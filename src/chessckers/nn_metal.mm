// Metal (MPSGraph) NN backend — Phase 6a/6b. See nn_metal.h.
#include "nn_metal.h"

#include "nn.hpp"  // ChesskersNet / WeightStore (the CPU forward + parity oracle)

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <cmath>
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
    const ChesskersNet& net = *p_->net;
    for (int k = 0; k < K; ++k) {
        const float v = net.value_v2(Fs[k]);
        const int N = (int)moves_per[k].size();
        std::vector<float> priors(N);
        if (N == 0) { out[k] = {v, priors}; continue; }
        const auto logits = net.policy_logits_v2(Fs[k], moves_per[k]);
        const float mx = *std::max_element(logits.begin(), logits.end());
        double s = 0.0;
        for (float l : logits) s += std::exp(l - mx);
        for (int i = 0; i < N; ++i) priors[i] = (float)(std::exp(logits[i] - mx) / s);
        out[k] = {v, priors};
    }
    return out;
}

}  // namespace cc
