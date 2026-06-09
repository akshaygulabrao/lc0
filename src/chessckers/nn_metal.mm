// Metal (MPSGraph) NN backend — Phase 6a/6b. See nn_metal.h.
#include "nn_metal.h"

#include "nn.hpp"  // ChesskersNet / WeightStore (the CPU forward + parity oracle)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

namespace cc {

float metal_matmul_selftest(int M, int K, int N) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) return -1.0f;  // no Metal GPU (e.g. headless CI)
        id<MTLCommandQueue> q = [dev newCommandQueue];

        std::vector<float> A((size_t)M * K), B((size_t)K * N);
        for (size_t i = 0; i < A.size(); ++i) A[i] = std::sin(0.1 * (double)i);
        for (size_t i = 0; i < B.size(); ++i) B[i] = std::cos(0.07 * (double)i);

        // CPU reference (the parity oracle).
        std::vector<float> ref((size_t)M * N, 0.0f);
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k) {
                const float a = A[(size_t)m * K + k];
                for (int n = 0; n < N; ++n) ref[(size_t)m * N + n] += a * B[(size_t)k * N + n];
            }

        MPSGraph* g = [MPSGraph new];
        MPSGraphTensor* ta = [g placeholderWithShape:@[ @(M), @(K) ]
                                            dataType:MPSDataTypeFloat32
                                                name:@"A"];
        MPSGraphTensor* tb = [g placeholderWithShape:@[ @(K), @(N) ]
                                            dataType:MPSDataTypeFloat32
                                                name:@"B"];
        MPSGraphTensor* tc = [g matrixMultiplicationWithPrimaryTensor:ta secondaryTensor:tb name:nil];

        id<MTLBuffer> ba = [dev newBufferWithBytes:A.data()
                                            length:A.size() * sizeof(float)
                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> bb = [dev newBufferWithBytes:B.data()
                                            length:B.size() * sizeof(float)
                                           options:MTLResourceStorageModeShared];
        MPSGraphTensorData* da = [[MPSGraphTensorData alloc] initWithMTLBuffer:ba
                                                                        shape:@[ @(M), @(K) ]
                                                                     dataType:MPSDataTypeFloat32];
        MPSGraphTensorData* db = [[MPSGraphTensorData alloc] initWithMTLBuffer:bb
                                                                        shape:@[ @(K), @(N) ]
                                                                     dataType:MPSDataTypeFloat32];

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* res =
            [g runWithMTLCommandQueue:q
                                feeds:@{ta : da, tb : db}
                        targetTensors:@[ tc ]
                     targetOperations:nil];
        MPSGraphTensorData* dc = res[tc];

        std::vector<float> out((size_t)M * N);
        [[dc mpsndarray] readBytes:out.data() strideBytes:nil];

        float md = 0.0f;
        for (size_t i = 0; i < out.size(); ++i) md = std::max(md, std::fabs(out[i] - ref[i]));
        return md;
    }
}

// ===================== Phase 6b: MPSGraph V2 spatial trunk =====================

namespace {

// A constant tensor straight off a WeightStore buffer (copies into NSData).
MPSGraphTensor* constFrom(MPSGraph* g, const std::vector<float>& v, NSArray<NSNumber*>* shape) {
    NSData* d = [NSData dataWithBytes:v.data() length:v.size() * sizeof(float)];
    return [g constantWithData:d shape:shape dataType:MPSDataTypeFloat32];
}

// conv k3p1, bias=false. weight is PyTorch [Cout,Cin,3,3] (== OIHW); data is NCHW.
MPSGraphTensor* convOp(MPSGraph* g, MPSGraphTensor* x, const std::vector<float>& w, int Cout,
                       int Cin) {
    MPSGraphTensor* wc = constFrom(g, w, @[ @(Cout), @(Cin), @3, @3 ]);
    MPSGraphConvolution2DOpDescriptor* d = [MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:1
                      strideInY:1
                dilationRateInX:1
                dilationRateInY:1
                         groups:1
                    paddingLeft:1
                   paddingRight:1
                     paddingTop:1
                  paddingBottom:1
                   paddingStyle:MPSGraphPaddingStyleExplicit
                     dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                  weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    return [g convolution2DWithSourceTensor:x weightsTensor:wc descriptor:d name:nil];
}

// GroupNorm over [N,C,H,W], G groups, eps 1e-5; gamma/beta are [C].
MPSGraphTensor* groupNorm(MPSGraph* g, MPSGraphTensor* x, int C, int G, int HW,
                          const std::vector<float>& gamma, const std::vector<float>& beta) {
    const int Cg = C / G, H = 10, W = 10;
    (void)HW;
    MPSGraphTensor* xr = [g reshapeTensor:x withShape:@[ @(-1), @(G), @(Cg), @(H), @(W) ] name:nil];
    NSArray<NSNumber*>* ax = @[ @2, @3, @4 ];
    MPSGraphTensor* mean = [g meanOfTensor:xr axes:ax name:nil];
    MPSGraphTensor* dif = [g subtractionWithPrimaryTensor:xr secondaryTensor:mean name:nil];
    MPSGraphTensor* var = [g meanOfTensor:[g squareWithTensor:dif name:nil] axes:ax name:nil];
    MPSGraphTensor* eps = [g constantWithScalar:1e-5 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* den =
        [g squareRootWithTensor:[g additionWithPrimaryTensor:var secondaryTensor:eps name:nil]
                           name:nil];
    MPSGraphTensor* norm = [g divisionWithPrimaryTensor:dif secondaryTensor:den name:nil];
    MPSGraphTensor* normc = [g reshapeTensor:norm withShape:@[ @(-1), @(C), @(H), @(W) ] name:nil];
    MPSGraphTensor* gc = constFrom(g, gamma, @[ @1, @(C), @1, @1 ]);
    MPSGraphTensor* bc = constFrom(g, beta, @[ @1, @(C), @1, @1 ]);
    return [g additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:normc
                                                          secondaryTensor:gc
                                                                     name:nil]
                        secondaryTensor:bc
                                   name:nil];
}

// Linear over the last dim of a token-major tensor [..., in]: x @ W^T + b, W is [out,in].
MPSGraphTensor* linearLast(MPSGraph* g, MPSGraphTensor* x, const std::vector<float>& W,
                           const std::vector<float>& b, int out, int in) {
    MPSGraphTensor* Wc = constFrom(g, W, @[ @(out), @(in) ]);
    MPSGraphTensor* Wt = [g transposeTensor:Wc dimension:0 withDimension:1 name:nil];  // [in,out]
    MPSGraphTensor* y = [g matrixMultiplicationWithPrimaryTensor:x secondaryTensor:Wt name:nil];
    MPSGraphTensor* bc = constFrom(g, b, @[ @1, @1, @(out) ]);
    return [g additionWithPrimaryTensor:y secondaryTensor:bc name:nil];
}

// LayerNorm over the last dim (size D) of a token-major tensor [N,T,D]; gamma/beta are [D].
MPSGraphTensor* layerNormLast(MPSGraph* g, MPSGraphTensor* x, int D,
                              const std::vector<float>& gamma, const std::vector<float>& beta) {
    NSArray<NSNumber*>* ax = @[ @2 ];
    MPSGraphTensor* mean = [g meanOfTensor:x axes:ax name:nil];
    MPSGraphTensor* dif = [g subtractionWithPrimaryTensor:x secondaryTensor:mean name:nil];
    MPSGraphTensor* var = [g meanOfTensor:[g squareWithTensor:dif name:nil] axes:ax name:nil];
    MPSGraphTensor* eps = [g constantWithScalar:1e-5 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* den =
        [g squareRootWithTensor:[g additionWithPrimaryTensor:var secondaryTensor:eps name:nil]
                           name:nil];
    MPSGraphTensor* norm = [g divisionWithPrimaryTensor:dif secondaryTensor:den name:nil];
    MPSGraphTensor* gc = constFrom(g, gamma, @[ @1, @1, @(D) ]);
    MPSGraphTensor* bc = constFrom(g, beta, @[ @1, @1, @(D) ]);
    return [g additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:norm
                                                          secondaryTensor:gc
                                                                     name:nil]
                        secondaryTensor:bc
                                   name:nil];
}

// Exact GELU: 0.5*x*(1+erf(x/sqrt2)) — matches nn.GELU() default / the CPU gelu_.
MPSGraphTensor* geluExact(MPSGraph* g, MPSGraphTensor* x) {
    MPSGraphTensor* inv2 = [g constantWithScalar:0.70710678118654752440 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* e = [g erfWithTensor:[g multiplicationWithPrimaryTensor:x
                                                           secondaryTensor:inv2
                                                                      name:nil]
                                    name:nil];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* half = [g constantWithScalar:0.5 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* s = [g additionWithPrimaryTensor:e secondaryTensor:one name:nil];
    return [g multiplicationWithPrimaryTensor:[g multiplicationWithPrimaryTensor:x
                                                               secondaryTensor:half
                                                                          name:nil]
                             secondaryTensor:s
                                        name:nil];
}

// One pre-norm Transformer block over the 100 square-tokens, mirroring transformer_block_.
// x is channel-major spatial [N,C,10,10]; returns the same layout.
MPSGraphTensor* transformerBlock(MPSGraph* g, MPSGraphTensor* x, int C, int Hn, const WeightStore& w,
                                 const std::string& p) {
    const int T = 100, hd = C / Hn;
    MPSGraphTensor* xf = [g reshapeTensor:x withShape:@[ @(-1), @(C), @(T) ] name:nil];
    MPSGraphTensor* t = [g transposeTensor:xf dimension:1 withDimension:2 name:nil];  // [N,T,C]

    // ---- self-attention (pre-norm) ----
    MPSGraphTensor* n =
        layerNormLast(g, t, C, w.at(p + "norm1.weight"), w.at(p + "norm1.bias"));
    MPSGraphTensor* qkv =
        linearLast(g, n, w.at(p + "attn.in_proj_weight"), w.at(p + "attn.in_proj_bias"), 3 * C, C);
    NSArray<NSNumber*>* qsh = @[ @(-1), @(T), @(Hn), @(hd) ];
    MPSGraphTensor* Q =
        [g sliceTensor:qkv dimension:2 start:0 length:C name:nil];
    MPSGraphTensor* Kt =
        [g sliceTensor:qkv dimension:2 start:C length:C name:nil];
    MPSGraphTensor* V =
        [g sliceTensor:qkv dimension:2 start:2 * C length:C name:nil];
    // [N,T,C] -> [N,Hn,T,hd]
    Q = [g transposeTensor:[g reshapeTensor:Q withShape:qsh name:nil] dimension:1 withDimension:2 name:nil];
    Kt = [g transposeTensor:[g reshapeTensor:Kt withShape:qsh name:nil] dimension:1 withDimension:2 name:nil];
    V = [g transposeTensor:[g reshapeTensor:V withShape:qsh name:nil] dimension:1 withDimension:2 name:nil];
    MPSGraphTensor* Ktt = [g transposeTensor:Kt dimension:2 withDimension:3 name:nil];  // [N,Hn,hd,T]
    MPSGraphTensor* scores = [g matrixMultiplicationWithPrimaryTensor:Q secondaryTensor:Ktt name:nil];
    MPSGraphTensor* scl = [g constantWithScalar:1.0 / std::sqrt((double)hd) dataType:MPSDataTypeFloat32];
    scores = [g multiplicationWithPrimaryTensor:scores secondaryTensor:scl name:nil];
    scores = [g softMaxWithTensor:scores axis:3 name:nil];
    MPSGraphTensor* O = [g matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:V name:nil];  // [N,Hn,T,hd]
    O = [g transposeTensor:O dimension:1 withDimension:2 name:nil];                                   // [N,T,Hn,hd]
    O = [g reshapeTensor:O withShape:@[ @(-1), @(T), @(C) ] name:nil];
    MPSGraphTensor* proj =
        linearLast(g, O, w.at(p + "attn.out_proj.weight"), w.at(p + "attn.out_proj.bias"), C, C);
    t = [g additionWithPrimaryTensor:t secondaryTensor:proj name:nil];

    // ---- feed-forward (pre-norm) ----
    MPSGraphTensor* n2 = layerNormLast(g, t, C, w.at(p + "norm2.weight"), w.at(p + "norm2.bias"));
    const int ff = (int)w.at(p + "ff.0.bias").size();
    MPSGraphTensor* h1 = linearLast(g, n2, w.at(p + "ff.0.weight"), w.at(p + "ff.0.bias"), ff, C);
    h1 = geluExact(g, h1);
    MPSGraphTensor* h2 = linearLast(g, h1, w.at(p + "ff.2.weight"), w.at(p + "ff.2.bias"), C, ff);
    t = [g additionWithPrimaryTensor:t secondaryTensor:h2 name:nil];

    MPSGraphTensor* back = [g transposeTensor:t dimension:1 withDimension:2 name:nil];  // [N,C,T]
    return [g reshapeTensor:back withShape:@[ @(-1), @(C), @10, @10 ] name:nil];
}

}  // namespace

struct MetalTrunkV2::Impl {
    int c_in = 16, c_filters = 96;
    bool ok = false;
    const ChesskersNet* net = nullptr;  // for the CPU value/gather heads (6c)
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> q = nil;
    MPSGraph* g = nil;
    MPSGraphTensor* in = nil;
    MPSGraphTensor* out = nil;
    // GPU value+policy heads (V2): a second cached graph fed the trunk's feature map F.
    bool heads_ok = false;
    int d_hidden = 256;
    MPSGraph* gh = nil;
    MPSGraphTensor* hF = nil;     // F placeholder [-1,C,10,10]
    MPSGraphTensor* hwdl = nil;   // value WDL logits [-1,1,3]
    int n_typ = 12;              // trailing move type scalars (d_move-102)
    MPSGraphTensor* hgfrom = nil;   // [-1] int32: board*100+from
    MPSGraphTensor* hgto = nil;     // [-1] int32: board*100+to
    MPSGraphTensor* hgboard = nil;  // [-1] int32: board index (path-mean gather)
    MPSGraphTensor* hpmask = nil;   // [-1,100] path mask
    MPSGraphTensor* hpdenom = nil;  // [-1,1] path-mean denominator
    MPSGraphTensor* hptyp = nil;    // [-1,n_typ] type scalars
    MPSGraphTensor* hlogits = nil;  // [-1] policy logits over the flattened M moves
};

MetalTrunkV2::MetalTrunkV2(const ChesskersNet& net) : p_(std::make_unique<Impl>()) {
    @autoreleasepool {
        p_->c_in = net.c_in;
        p_->c_filters = net.c_filters;
        p_->net = &net;
        p_->dev = MTLCreateSystemDefaultDevice();
        if (!p_->dev) return;  // no GPU
        p_->q = [p_->dev newCommandQueue];
        const auto& w = net.w;
        const int C = net.c_filters, HW = 100;
        MPSGraph* g = [MPSGraph new];
        p_->g = g;
        MPSGraphTensor* x = [g placeholderWithShape:@[ @(-1), @(net.c_in), @10, @10 ]
                                           dataType:MPSDataTypeFloat32
                                               name:@"pos"];
        p_->in = x;
        x = convOp(g, x, w.at("position_trunk.0.weight"), C, net.c_in);
        x = groupNorm(g, x, C, 8, HW, w.at("position_trunk.1.weight"), w.at("position_trunk.1.bias"));
        x = [g reLUWithTensor:x name:nil];
        for (int k = 3;; ++k) {
            const std::string p = "position_trunk." + std::to_string(k) + ".";
            if (w.tensors.count(p + "pos")) {
                MPSGraphTensor* pe = constFrom(g, w.at(p + "pos"), @[ @1, @(C), @10, @10 ]);
                x = [g additionWithPrimaryTensor:x secondaryTensor:pe name:nil];
            } else if (w.tensors.count(p + "conv1.weight")) {
                MPSGraphTensor* c1 = convOp(g, x, w.at(p + "conv1.weight"), C, C);
                c1 = groupNorm(g, c1, C, 8, HW, w.at(p + "bn1.weight"), w.at(p + "bn1.bias"));
                c1 = [g reLUWithTensor:c1 name:nil];
                MPSGraphTensor* c2 = convOp(g, c1, w.at(p + "conv2.weight"), C, C);
                c2 = groupNorm(g, c2, C, 8, HW, w.at(p + "bn2.weight"), w.at(p + "bn2.bias"));
                c2 = [g additionWithPrimaryTensor:c2 secondaryTensor:x name:nil];
                x = [g reLUWithTensor:c2 name:nil];
            } else if (w.tensors.count(p + "attn.in_proj_weight")) {
                x = transformerBlock(g, x, C, net.n_heads, w, p);
            } else {
                break;
            }
        }
        p_->out = x;
        p_->ok = true;

        // --- GPU heads (V2): a second graph, fed F, outputs the value WDL logits. The policy
        // head (6e) is added to this graph next; until then priors stay on the CPU. ---
        if (net.is_v2) {
            const int dh = net.d_hidden;
            p_->d_hidden = dh;
            MPSGraph* gh = [MPSGraph new];
            p_->gh = gh;
            MPSGraphTensor* F = [gh placeholderWithShape:@[ @(-1), @(C), @10, @10 ]
                                                dataType:MPSDataTypeFloat32
                                                    name:@"F"];
            p_->hF = F;
            MPSGraphTensor* pooled = [gh meanOfTensor:F axes:@[ @2, @3 ] name:nil];      // [K,C,1,1]
            pooled = [gh reshapeTensor:pooled withShape:@[ @(-1), @1, @(C) ] name:nil];  // [K,1,C]
            MPSGraphTensor* vt = linearLast(gh, pooled, w.at("value_trunk.0.weight"),
                                            w.at("value_trunk.0.bias"), dh, C);
            vt = layerNormLast(gh, vt, dh, w.at("value_trunk.1.weight"), w.at("value_trunk.1.bias"));
            vt = [gh reLUWithTensor:vt name:nil];
            MPSGraphTensor* v1 = linearLast(gh, vt, w.at("value_head.0.weight"),
                                            w.at("value_head.0.bias"), dh / 2, dh);
            v1 = layerNormLast(gh, v1, dh / 2, w.at("value_head.1.weight"), w.at("value_head.1.bias"));
            v1 = [gh reLUWithTensor:v1 name:nil];
            p_->hwdl = linearLast(gh, v1, w.at("value_head.3.weight"), w.at("value_head.3.bias"),
                                  3, dh / 2);  // [K,1,3]

            // ---- policy (gather) head, flattened over the M moves of all boards ----
            const int ntyp = net.d_move - 102;
            p_->n_typ = ntyp;
            MPSGraphTensor* gfrom = [gh placeholderWithShape:@[ @(-1) ] dataType:MPSDataTypeInt32 name:@"gfrom"];
            MPSGraphTensor* gto = [gh placeholderWithShape:@[ @(-1) ] dataType:MPSDataTypeInt32 name:@"gto"];
            MPSGraphTensor* gboard = [gh placeholderWithShape:@[ @(-1) ] dataType:MPSDataTypeInt32 name:@"gboard"];
            MPSGraphTensor* pmask = [gh placeholderWithShape:@[ @(-1), @100 ] dataType:MPSDataTypeFloat32 name:@"pmask"];
            MPSGraphTensor* pdenom = [gh placeholderWithShape:@[ @(-1), @1 ] dataType:MPSDataTypeFloat32 name:@"pdenom"];
            MPSGraphTensor* ptyp = [gh placeholderWithShape:@[ @(-1), @(ntyp) ] dataType:MPSDataTypeFloat32 name:@"ptyp"];
            p_->hgfrom = gfrom; p_->hgto = gto; p_->hgboard = gboard;
            p_->hpmask = pmask; p_->hpdenom = pdenom; p_->hptyp = ptyp;

            // F -> token-major [K,100,C] (and flat [K*100,C]) for the gathers.
            MPSGraphTensor* Fc = [gh reshapeTensor:F withShape:@[ @(-1), @(C), @100 ] name:nil];    // [K,C,100]
            MPSGraphTensor* Ftok = [gh transposeTensor:Fc dimension:1 withDimension:2 name:nil];    // [K,100,C]
            MPSGraphTensor* Fflat = [gh reshapeTensor:Ftok withShape:@[ @(-1), @(C) ] name:nil];    // [K*100,C]

            // endpoint gathers -> [M,1,C]
            MPSGraphTensor* FF = [gh gatherWithUpdatesTensor:Fflat indicesTensor:gfrom axis:0 batchDimensions:0 name:nil];
            MPSGraphTensor* TF = [gh gatherWithUpdatesTensor:Fflat indicesTensor:gto axis:0 batchDimensions:0 name:nil];
            FF = [gh reshapeTensor:FF withShape:@[ @(-1), @1, @(C) ] name:nil];
            TF = [gh reshapeTensor:TF withShape:@[ @(-1), @1, @(C) ] name:nil];

            // path-mean PF = (pmask @ F_board) / denom -> [M,1,C]
            MPSGraphTensor* Fb = [gh gatherWithUpdatesTensor:Ftok indicesTensor:gboard axis:0 batchDimensions:0 name:nil];  // [M,100,C]
            MPSGraphTensor* pm = [gh reshapeTensor:pmask withShape:@[ @(-1), @1, @100 ] name:nil];                          // [M,1,100]
            MPSGraphTensor* PF = [gh matrixMultiplicationWithPrimaryTensor:pm secondaryTensor:Fb name:nil];                 // [M,1,C]
            PF = [gh divisionWithPrimaryTensor:PF
                               secondaryTensor:[gh reshapeTensor:pdenom withShape:@[ @(-1), @1, @1 ] name:nil]
                                          name:nil];

            // projections + ctx MLP + scaled dot
            MPSGraphTensor* src = linearLast(gh, FF, w.at("src_proj.weight"), w.at("src_proj.bias"), dh, C);
            MPSGraphTensor* tgt = linearLast(gh, TF, w.at("tgt_proj.weight"), w.at("tgt_proj.bias"), dh, C);
            MPSGraphTensor* typ3 = [gh reshapeTensor:ptyp withShape:@[ @(-1), @1, @(ntyp) ] name:nil];
            MPSGraphTensor* ctxin = [gh concatTensors:@[ FF, TF, PF, typ3 ] dimension:2 name:nil];  // [M,1,3C+ntyp]
            MPSGraphTensor* h = linearLast(gh, ctxin, w.at("ctx_mlp.0.weight"), w.at("ctx_mlp.0.bias"),
                                           dh, 3 * C + ntyp);
            h = layerNormLast(gh, h, dh, w.at("ctx_mlp.1.weight"), w.at("ctx_mlp.1.bias"));
            h = [gh reLUWithTensor:h name:nil];
            MPSGraphTensor* ctx = linearLast(gh, h, w.at("ctx_mlp.3.weight"), w.at("ctx_mlp.3.bias"), 1, dh);  // [M,1,1]

            MPSGraphTensor* dot = [gh reductionSumWithTensor:[gh multiplicationWithPrimaryTensor:src
                                                                                secondaryTensor:tgt
                                                                                           name:nil]
                                                       axis:2
                                                       name:nil];
            dot = [gh reshapeTensor:dot withShape:@[ @(-1), @1, @1 ] name:nil];  // [M,1,1]
            MPSGraphTensor* scl =
                [gh constantWithScalar:1.0 / std::sqrt((double)dh) dataType:MPSDataTypeFloat32];
            dot = [gh multiplicationWithPrimaryTensor:dot secondaryTensor:scl name:nil];
            MPSGraphTensor* logit = [gh additionWithPrimaryTensor:dot secondaryTensor:ctx name:nil];  // [M,1,1]
            p_->hlogits = [gh reshapeTensor:logit withShape:@[ @(-1) ] name:nil];  // [M]

            // CC_CPU_HEADS=1 forces the CPU value/gather heads (the pre-GPU-heads path) — for
            // A/B benchmarking and as an escape hatch if a GPU-head issue ever surfaces.
            p_->heads_ok = (std::getenv("CC_CPU_HEADS") == nullptr);
        }
    }
}

MetalTrunkV2::~MetalTrunkV2() = default;

bool MetalTrunkV2::ok() const { return p_ && p_->ok; }

std::vector<std::vector<float>> MetalTrunkV2::run(
    const std::vector<std::vector<float>>& positions) const {
    const int K = (int)positions.size();
    std::vector<std::vector<float>> result;
    if (!p_->ok || K == 0) return result;
    @autoreleasepool {
        const int Cin = p_->c_in, C = p_->c_filters, HW = 100;
        std::vector<float> flat((size_t)K * Cin * HW);
        for (int k = 0; k < K; ++k)
            std::copy(positions[k].begin(), positions[k].end(), &flat[(size_t)k * Cin * HW]);
        id<MTLBuffer> buf = [p_->dev newBufferWithBytes:flat.data()
                                                 length:flat.size() * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        MPSGraphTensorData* td =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:buf
                                                    shape:@[ @(K), @(Cin), @10, @10 ]
                                                 dataType:MPSDataTypeFloat32];
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* res =
            [p_->g runWithMTLCommandQueue:p_->q
                                    feeds:@{p_->in : td}
                            targetTensors:@[ p_->out ]
                         targetOperations:nil];
        std::vector<float> outf((size_t)K * C * HW);
        [[res[p_->out] mpsndarray] readBytes:outf.data() strideBytes:nil];
        result.resize(K);
        for (int k = 0; k < K; ++k)
            result[k].assign(&outf[(size_t)k * C * HW], &outf[(size_t)(k + 1) * C * HW]);
    }
    return result;
}

std::vector<std::pair<float, std::vector<float>>> MetalTrunkV2::eval_batch(
    const std::vector<std::vector<float>>& positions,
    const std::vector<std::vector<std::vector<float>>>& moves_per) const {
    const int K = (int)positions.size();
    std::vector<std::pair<float, std::vector<float>>> out(K);
    if (!p_->ok || !p_->net) return out;
    const auto Fs = run(positions);  // GPU trunk (cached graph)
    if (p_->heads_ok) return eval_heads_from_F(Fs, moves_per);  // GPU value+policy heads
    // Fallback (no V2 head graph): CPU value/gather heads.
    const ChesskersNet& net = *p_->net;
    for (int k = 0; k < K; ++k) {
        const float v = net.value_v2(Fs[k]);
        const int N = (int)moves_per[k].size();
        if (N == 0) { out[k] = {v, std::vector<float>()}; continue; }
        const auto logits = net.policy_logits_v2(Fs[k], moves_per[k]);
        out[k] = {v, softmax_priors(logits.data(), N)};
    }
    return out;
}

// Run ONLY the heads (value + policy) on already-computed trunk features Fs, entirely on the
// GPU (the cached head graph): F + the flattened move data are fed in, WDL logits + the M
// policy logits come back, and only the two softmaxes run on the host (bit-identical to the
// oracle's). Exposed so a parity test can feed the SAME F here and to the CPU oracle,
// isolating the head port from the trunk's float drift.
std::vector<std::pair<float, std::vector<float>>> MetalTrunkV2::eval_heads_from_F(
    const std::vector<std::vector<float>>& Fs,
    const std::vector<std::vector<std::vector<float>>>& moves_per) const {
    const int K = (int)Fs.size();
    std::vector<std::pair<float, std::vector<float>>> out(K);
    if (!p_->heads_ok || K == 0) return out;
    const ChesskersNet& net = *p_->net;
    @autoreleasepool {
        const int C = p_->c_filters, HW = 100;
        auto mkData = [&](const void* bytes, size_t len, NSArray<NSNumber*>* shape,
                          MPSDataType dt) -> MPSGraphTensorData* {
            id<MTLBuffer> b = [p_->dev newBufferWithBytes:bytes
                                                   length:len
                                                  options:MTLResourceStorageModeShared];
            return [[MPSGraphTensorData alloc] initWithMTLBuffer:b shape:shape dataType:dt];
        };
        std::vector<float> flat((size_t)K * C * HW);
        for (int k = 0; k < K; ++k)
            std::copy(Fs[k].begin(), Fs[k].end(), &flat[(size_t)k * C * HW]);
        MPSGraphTensorData* dF = mkData(flat.data(), flat.size() * sizeof(float),
                                        @[ @(K), @(C), @10, @10 ], MPSDataTypeFloat32);

        const FlatMoves fm = flatten_moves(moves_per, net.d_move);
        const int M = fm.M, ntyp = fm.n_typ;
        std::vector<float> wdl((size_t)K * 3), logits;

        if (M > 0) {
            std::vector<int32_t> gfrom(M), gto(M);
            for (int i = 0; i < M; ++i) {
                gfrom[i] = fm.board_of[i] * 100 + fm.from_idx[i];
                gto[i] = fm.board_of[i] * 100 + fm.to_idx[i];
            }
            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
                p_->hF : dF,
                p_->hgfrom : mkData(gfrom.data(), (size_t)M * 4, @[ @(M) ], MPSDataTypeInt32),
                p_->hgto : mkData(gto.data(), (size_t)M * 4, @[ @(M) ], MPSDataTypeInt32),
                p_->hgboard : mkData(fm.board_of.data(), (size_t)M * 4, @[ @(M) ], MPSDataTypeInt32),
                p_->hpmask : mkData(fm.pathmask.data(), (size_t)M * 100 * 4, @[ @(M), @100 ], MPSDataTypeFloat32),
                p_->hpdenom : mkData(fm.denom.data(), (size_t)M * 4, @[ @(M), @1 ], MPSDataTypeFloat32),
                p_->hptyp : mkData(fm.typ.data(), (size_t)M * ntyp * 4, @[ @(M), @(ntyp) ], MPSDataTypeFloat32),
            };
            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* res =
                [p_->gh runWithMTLCommandQueue:p_->q
                                         feeds:feeds
                                 targetTensors:@[ p_->hwdl, p_->hlogits ]
                              targetOperations:nil];
            [[res[p_->hwdl] mpsndarray] readBytes:wdl.data() strideBytes:nil];
            logits.resize(M);
            [[res[p_->hlogits] mpsndarray] readBytes:logits.data() strideBytes:nil];
        } else {
            NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* res =
                [p_->gh runWithMTLCommandQueue:p_->q
                                         feeds:@{p_->hF : dF}
                                 targetTensors:@[ p_->hwdl ]
                              targetOperations:nil];
            [[res[p_->hwdl] mpsndarray] readBytes:wdl.data() strideBytes:nil];
        }

        for (int k = 0; k < K; ++k) {
            const float* z = &wdl[(size_t)k * 3];
            const float mx = std::max({z[0], z[1], z[2]});
            const double e0 = std::exp(z[0] - mx), e1 = std::exp(z[1] - mx), e2 = std::exp(z[2] - mx);
            const float v = static_cast<float>((e0 - e2) / (e0 + e1 + e2));
            const int lo = fm.board_off[k], n = fm.board_off[k + 1] - lo;
            out[k] = {v, (M > 0 && n > 0) ? softmax_priors(&logits[lo], n) : std::vector<float>()};
        }
    }
    return out;
}

}  // namespace cc
