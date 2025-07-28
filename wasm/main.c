#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <emscripten/emscripten.h>


#define ARENA_BYTES (1u << 30)
#define ENERGY_SIZE 64

// NOTE: ensure 12 * denom < MAX_INT64
#define NORM_DENOM 100000000

#define Getter(name,type) \
EMSCRIPTEN_KEEPALIVE \
type Get_##name(void) {return name;}

#define Assert(cond, msg) if (!(cond)) {printf(msg); *(volatile int*)0 = 0;}

#define ShapesEqual(A, B) A->rows == B->rows && A->cols == B->cols

typedef uint8_t byte;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int32_t i32;
typedef int64_t i64;

typedef float f32;

typedef struct {
    f32 x;
    f32 y;
} Point;

typedef struct {
    byte* base;
    u64 size;
    u64 pos;
} Arena;


typedef struct {
    int rows;
    int cols;
    float* data;
} Mat;


typedef struct {
    int nLayers;
    Mat* weights;
    Mat* biases;
    Mat* wGrad;
    Mat* bGrad;
} Net;

typedef struct {
    Arena arena;
    f32* energy;
    Mat* energySamples;
    byte* pixels;

    Net* ebmNet;
    byte* ebmPixels;
    Mat* ebmInters;
    Mat* ebmInterGrads;
    Mat* ebmTrainOuts;
    Mat* ebmSamples;
    f32* ebmEnergy;
} AppState;



///--- BEGIN CONSTANTS GETTERS ---///
Getter(ENERGY_SIZE, u32)

///--- END CONSTANTS GETTERS ---///


byte* arena_alloc(Arena* arena, u64 size) {
    if (arena->pos + size >= arena->size) {
        printf("ARENA OVERFLOW! Trying to alloc: %lld on arena with %lld/%lld taken\n", 
            size,
            arena->pos,
            arena->size
        );
        return 0;
    }
    byte* res = arena->base + arena->pos;
    arena->pos += size;
    return res;
}


void copy_bytes(byte* src, byte* dest, int n) {
    for (int i=0; i<n; i++) {
        dest[i] = src[i];
    }
}

void copy_f32(f32* src, f32* dest, int n) {
    for (int i=0; i < n; i++) {
        dest[i] = src[i];
    }
}

f32 rand_normal() {
    f32 u1 = (rand() + 1.0) / ((f32) RAND_MAX + 2.0);
    f32 u2 = (rand() + 1.0) / ((f32) RAND_MAX + 2.0);
    f32 r = sqrtf(-2 * logf(u1));
    f32 th = 2.0 * M_PI * u2;
    return r * cosf(th);
}


///--- BEGIN MATRIX ---///



Mat* alloc_mat(int rows, int cols, Arena* arena) {
    byte* resPtr = arena_alloc(arena, sizeof(Mat) + sizeof(f32) * rows * cols);
    Mat* res = (Mat*) resPtr;
    res->rows = rows;
    res->cols = cols;
    res->data = (f32*) (resPtr + sizeof(Mat));
    return res;
}


void rand_mat_(Mat* mat, float min, float max) {
    for (int i=0; i < mat->rows; i++) {
        for (int j=0; j < mat->cols; j++) {
            mat->data[i * mat->cols + j] = (float)rand() / (float)((float)RAND_MAX / (max-min)) + min;
        }
    }
}

void zero_mat_(Mat* mat) {
    for (int i=0; i < mat->rows; i++) {
        for (int j=0; j < mat->cols; j++) {
            mat->data[i * mat->cols + j] = 0;
        }
    }
}


void print_mat(Mat* mat) {
    printf("Mat: (%d x %d)\n", mat->rows, mat->cols);
    for (int i=0; i < mat->rows; i++) {
        for (int j=0; j < mat->cols; j++) {
            printf("%.03f ", mat->data[i * mat->cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}


void matmul(Mat* a, Mat* b, Mat* res) {
    int aRows = a->rows;
    int bRows = b->rows;
    int aCols = a->cols;
    int bCols = b->cols;
    int resCols = bCols;

    f32* aData = a->data;
    f32* bData = b->data;
    f32* resData = res->data;
    if (aCols != bRows) {
        printf("ERROR A COLS AND B ROWS DO NOT MATCH IN MATMUL\n");
        return;
    }


    for (int i=0; i < aRows; i++) {
        for (int j=0; j < bCols; j++) {
            float resVal = 0;
            for (int k=0; k < aCols; k++) {
                float aVal = aData[i * aCols + k]; 
                float bVal = bData[k * bCols + j];
                resVal += aVal * bVal;
            }
            resData[i*resCols + j] = resVal;
        }
    }
}

void matmul_grad_(Mat* a, Mat* b, Mat* out_grad, Mat* res_a, Mat* res_b) {
    // grad a = G @ B_T
    //    nxk  nxm  mxk
    // ein: G,B "ij,lj->il"
    // grad b = A_T @ G
    //    kxm   kxn   nxm
    // ein: A,G "il,ij->lj"
    int n = a->rows;
    int k = a->cols;
    int m = b->cols;

    Assert(a->rows == out_grad->rows, "ERROR: A ROWS DO NOT MATCH OUT_GRAD ROWS IN MATMUL_GRAD_\n");
    Assert(a->cols == b->rows, "ERROR: A COLS DO NOT MATCH B ROWS IN MATMUL_GRAD_\n");
    Assert(b->cols == out_grad->cols, "ERROR: B COLS DO NOT MATCH OUT_GRAD COLS IN MATMUL_GRAD_\n");
    Assert(b != res_b, "ERROR: b and res_b point to the same place in matmul_grad_. This will cause b to be overwritten before a_grad can be calculated.\n")



    // NOTE b grad (aka the weight grad in a network) is calculated first bc that way nothing
    // bad will happen if a and res_a are the same. a will be overwritten but only after it's values
    // were used. I expect this case with a (batch inputs) not b (weights) hence the ordering.
    for (int l=0; l < k; l++) {
        for (int j=0; j<m; j++) {
            f32* b_ptr = &res_b->data[l * m + j];
            *b_ptr = 0;
            for (int i=0; i<n; i++) {
                *b_ptr += out_grad->data[i * m + j] * a->data[i * k + l];
            }
        }
    }

    for (int i=0; i<n; i++) {
        for (int l=0; l < k; l++) {
            f32* a_ptr = &res_a->data[i * k + l];
            *a_ptr = 0;
            for (int j=0; j<m; j++) {
                *a_ptr += out_grad->data[i * m + j] * b->data[l * m + j];
            }
        }
    }
    
}

void matadd_bias_(Mat* a, Mat* b) {
    if (a->cols != b->cols) {
        printf("ERROR A COLS AND B COLS DO NOT MATCH IN MATADD_BIAS_\nA shape:(%d,%d) vs B shape: (%d,%d)\n", a->rows, a->cols, b->rows, b->cols);
        return;
    }
    int aRows = a->rows;
    int aCols = a->cols;
    f32* aData = a->data;
    f32* bData = b->data;
    for (int i=0; i<aRows; i++) {
        for (int j=0; j < aCols; j++) {
            aData[i*aCols + j] += bData[j];
        }
    }
}

void matadd_bias_grad_(Mat* out_grad, Mat* res) {
    Assert(out_grad->cols == res->cols, "ERROR BIAS AND OUT_GRAD HAVE DIFFERENT SHAPES IN BIAS_GRAD_\n")
    int cols = out_grad->cols; 

    for (int j=0; j < cols; j++) {
        res->data[j] = 0;
        for (int i=0; i < out_grad->rows; i++) {
            res->data[j] += out_grad->data[i * cols + j];
        }
    }
}


void sub_mat_(Mat* a, Mat* b) {
    Assert(ShapesEqual(a, b), "SHAPES DON'T MATCH IN SUB_MAT_\n");
    for (int i=0; i < a->rows * a->cols; i++) {
        a->data[i] -= b->data[i];
    }
}


void mat_const_mul_(Mat* m, f32 c) {
    for (int i=0; i < m->rows * m->cols; i++) {
        m->data[i] *= c;
    }
}


void clamp_mat_01_(Mat* m) {
    f32* data = m->data;
    for (int i=0; i < m->rows * m->cols; i++) {
        f32 val = *data;
        *data++ = val * (val >= 0. && val < 1.) +  (val >= 1);
    }
}

void relu_(Mat* m) {
    f32* data = m->data;
    for(int i=0; i < m->rows * m->cols; i++) {
        f32 val = *data;
        *data++ = (val >= 0) * val;
    }
}

void relu_grad_(Mat* relu_y, Mat* out_grad, Mat* res) {
    f32* y = relu_y->data;
    f32* resPtr = res->data;
    f32* out = out_grad->data;
    for (int i=0; i < relu_y->rows * relu_y->cols; i++) {
        *resPtr++ = (*y > 0) * (*out);
        y++;
        out++;
    }
}

void leaky_relu_(Mat* m) {
    f32* data = m->data;
    for (int i=0; i < m->rows * m->cols; i++) {
        f32 val = *data;
        *data++ = (val < 0) * 0.0625 * val + (val >= 0) * val;
    }
}


void leaky_relu_grad_(Mat* lrelu_y, Mat* out_grad, Mat* res) {
    f32* y = lrelu_y->data;
    f32* resPtr = res->data;
    f32* grad = out_grad->data;
    for (int i=0; i < lrelu_y->rows * lrelu_y->cols; i++) {
        *resPtr++ = ((*y < 0) * 0.0625 + (*y >= 0)) * (*grad);
        y++;
        grad++;
    }
}


void sigmoid_(Mat* m) {
    f32* data = m->data;
    for (int i=0; i < m->rows * m->cols; i++) {
        f32 val = *data;
        *data++ = 1. / (1 + exp(-val));
    }
}

void sigmoid_grad_(Mat* sig_y, Mat* out_grad, Mat* res) {
    Assert(ShapesEqual(sig_y, res), "SIG_Y AND RES SHAPES DON'T MATCH IN SIGMOID_GRAD\n");
    Assert(ShapesEqual(out_grad, res), "OUT_GRAD AND RES SHAPES DON'T MATCH IN SIGMOID_GRAD\n");
    f32* y = sig_y->data;
    f32* grad = out_grad->data;
    f32* resPtr = res->data;
    for (int i=0; i < sig_y->rows * sig_y->cols; i++) {
        *resPtr++ = (*y) * (1 - (*y)) * (*grad);
        y++;
        grad++;
    }
}
///--- END MATRIX ---///

///--- BEGIN NETWORK ---///

Net* alloc_net(int numDims, int* dims, Arena* arena) {
    int nLayers = numDims-1;
    // Struct size + weights + biases + grads per layer
    int headSize = sizeof(Net) + 4 * sizeof(Mat) * (nLayers);

    int dataSize = 0;
    for (int i = 0; i < nLayers; i++) {
        // weight + bias and grads
        dataSize += 2 * dims[i] * dims[i+1] + 2 * dims[i+1];
    }
    int totalSize = headSize + dataSize * sizeof(f32);

    byte* resPtr = arena_alloc(arena, totalSize);
    Net* res = (Net*) resPtr;
    res->weights = (Mat*) (resPtr + sizeof(Net));
    res->biases = (Mat*) (resPtr + sizeof(Net) + sizeof(Mat) * (nLayers));
    res->wGrad = (Mat*) (resPtr + sizeof(Net) + 2 * sizeof(Mat) * (nLayers));
    res->bGrad = (Mat*) (resPtr + sizeof(Net) + 3 * sizeof(Mat) * (nLayers));
    f32* nextData = (f32*) (resPtr + headSize);

    // Num dims includes inputs and outputs so subtract 1 for total layers
    res->nLayers = nLayers;

    for (int i=0; i < nLayers; i++) {
        res->weights[i].rows = dims[i];
        res->weights[i].cols = dims[i+1];
        res->weights[i].data = (f32*) nextData;
        nextData += dims[i] * dims[i+1];

        res->biases[i].rows = 1;
        res->biases[i].cols = dims[i+1];
        res->biases[i].data = (f32*) nextData;
        nextData += dims[i+1];

        // gradients for w
        res->wGrad[i].rows = dims[i];
        res->wGrad[i].cols = dims[i+1];
        res->wGrad[i].data = (f32*) nextData;
        nextData += dims[i] * dims[i+1];

        // gradients for b
        res->bGrad[i].rows = 1;
        res->bGrad[i].cols = dims[i+1];
        res->bGrad[i].data = (f32*) nextData;
        nextData += dims[i+1];
        
    }
    return res;
}

void print_net_shape(Net* net) {
    printf("\nNum layers: %d\n", net->nLayers);
    for (int i=0; i < net->nLayers; i++) {
        printf("weight %d: (%d, %d)\n", i, net->weights[i].rows, net->weights[i].cols);
        printf("bias %d: (%d, %d)\n", i, net->biases[i].rows, net->biases[i].cols);
    }
}

Mat* alloc_inters(int batchSize, Net* net, Arena* arena) {
    int dataSize = 0; 
    for (int i=0; i < net->nLayers; i++) {
        int ins = net->weights[i].rows;
        dataSize += batchSize * ins;
    }
    // output data dataSize
    dataSize += batchSize * net->weights[net->nLayers-1].cols;
    // mat headers
    int headSize = (net->nLayers+1) * sizeof(Mat);
    int totalSize = dataSize * sizeof(f32) + headSize;
    byte* resPtr = arena_alloc(arena, totalSize);
    Mat* res = (Mat*) resPtr;
    f32* nextData = (f32*)(resPtr + headSize);
    for (int i=0; i < net->nLayers; i++) {
        res[i].rows = batchSize;
        res[i].cols = net->weights[i].rows;
        res[i].data = nextData;
        nextData += res[i].rows * res[i].cols;
    }

    res[net->nLayers].rows = batchSize;
    res[net->nLayers].cols = net->weights[net->nLayers-1].cols;
    res[net->nLayers].data = nextData;
    return res;
} 

void rand_init(Net* net, f32 low, f32 high) {
    for (int i=0; i < net->nLayers; i++) {
        rand_mat_(&net->weights[i], low, high);
        rand_mat_(&net->biases[i], low, high);
    }
}

void xavier_init(Net* net) {
    for (int i=0; i < net->nLayers; i++) {
        f32 scale = 1/sqrtf(net->weights[i].rows);
        f32 low = -scale;
        f32 high = scale;
        rand_mat_(&net->weights[i], low, high);
        zero_mat_(&net->biases[i]);
    }
}

void zero_grads_(Net* net, Mat* interGrads) {
    for (int i=0; i < net->nLayers; i++) {
        zero_mat_(&net->wGrad[i]);
        zero_mat_(&net->bGrad[i]);
        zero_mat_(&interGrads[i]);
    }
}

void forward_net(Net* net, Mat* inters) {
    for (int i=0; i < net->nLayers-1; i++) {
        matmul(&inters[i], &net->weights[i], &inters[i+1]);
        matadd_bias_(&inters[i+1], &net->biases[i]);
        leaky_relu_(&inters[i+1]);
    }
    matmul(&inters[net->nLayers - 1], &net->weights[net->nLayers-1], &inters[net->nLayers]);
    matadd_bias_(&inters[net->nLayers], &net->biases[net->nLayers-1]);
}

f32 mean_sq_err(Mat* tar, Mat* out) {
    if (tar->rows != out->rows || tar->cols != out->cols) {
        printf("ERROR TARGET AND OUTPUT SHAPES DON'T MATCH IN MEAN_SQ_ERR:\n  (%d, %d) VS (%d, %d)\n",
            tar->rows, tar->cols, out->rows, out->cols
        );
    }
    int rows = tar->rows;
    int cols = tar->cols;
    f32 res = 0;
    int count = 0;
    for (int i =0; i < tar->rows * tar->cols; i++) {
        f32 diff = tar->data[i] - out->data[i];
        res += diff * diff;
        count++;
    }
    return res / count;
}


void mean_sq_grad_(Mat* tar, Mat* out, Mat* out_grad) {
    Assert(ShapesEqual(tar, out), "ERROR SHAPES NOT EQUAL IN MEAN_SQ_GRAD\n");
    int rows = tar->rows;
    int cols = tar->cols;
    int n = rows * cols;

    f32* a = out->data;
    f32* b = tar->data;
    f32* grad = out_grad->data;
    for (int i=0; i < rows * cols; i++) {
        *grad++ = 2. / n * ((*a++) - (*b++));
    }

}


void backward_net(Net* net, Mat* inters, Mat* interGrads) {
    for (int i=net->nLayers-1; i >= 0; i--) {

        // skip sigmoid for last layer
        if (i != net->nLayers-1) {
            // sigmoid
            leaky_relu_grad_(&inters[i+1], &interGrads[i+1], &interGrads[i+1]);
        }
        // bias
        matadd_bias_grad_(&interGrads[i+1], &net->bGrad[i]);
        // matmul
        matmul_grad_(&inters[i], &net->weights[i], &interGrads[i+1], &interGrads[i], &net->wGrad[i]);
    }
}


void train_step(Net* net, Mat* inters, Mat* interGrads, Mat* tar, f32 lr) {
    // Populate inters
    forward_net(net, inters);
    Mat* outs_ = &inters[net->nLayers];

    zero_grads_(net, interGrads);
    
    // overwrite inters for grad
    // TODO bug fixes
    mean_sq_grad_(tar, outs_, &interGrads[net->nLayers]);

    backward_net(net, inters, interGrads);

    // step in direction opposite grad
    for (int i=0; i < net->nLayers; i++) {
        mat_const_mul_(&net->wGrad[i], lr);
        sub_mat_(&net->weights[i], &net->wGrad[i]);
        mat_const_mul_(&net->bGrad[i], lr);
        sub_mat_(&net->biases[i], &net->bGrad[i]);
    }

}
///--- END NETWORK ---///


///--- BEGIN EBM ---///

void langevin_samples(f32* energy, Mat* samples, int nSteps, f32 stepSize, f32 energyScale) {
    f32 noiseScale = sqrtf(2 * stepSize);
    f32 gridScale = 1. / ENERGY_SIZE;

    for (int step=0; step < nSteps; step++) {
        for (int i=0; i < samples->rows * samples->cols; i+=2) {
            int xsam = (int)(samples->data[i] * ENERGY_SIZE);
            xsam = xsam * (xsam >= 0 && xsam < ENERGY_SIZE) + (ENERGY_SIZE - 1) * (xsam >= ENERGY_SIZE);
            int xlow = xsam - 1;        
            xlow = xlow * (xlow >= 0);
            int xhigh = xsam + 1;
            xhigh = xhigh * (xhigh < ENERGY_SIZE) + (ENERGY_SIZE - 1) * (xhigh >= ENERGY_SIZE);

            int ysam = (int)(samples->data[i+1] * ENERGY_SIZE);
            ysam = ysam * (ysam >= 0 && ysam < ENERGY_SIZE) + (ENERGY_SIZE - 1) * (ysam >= ENERGY_SIZE);
            int ylow =  ysam - 1;        
            ylow = ylow * (ylow >= 0);
            int yhigh = ysam + 1;
            yhigh = yhigh * (yhigh < ENERGY_SIZE) + (ENERGY_SIZE - 1) * (yhigh >= ENERGY_SIZE);

            f32 samEnergy = energy[ysam * ENERGY_SIZE + xsam];
            // loops? What are those?
            // scharr kernel to estimate gradient
            f32 xgrad = -3 * (energy[ylow * ENERGY_SIZE + xlow] - samEnergy) - 
                        10 * (energy[ysam * ENERGY_SIZE + xlow] - samEnergy) -
                        3 * (energy[yhigh * ENERGY_SIZE + xlow] - samEnergy) +
                        3 * (energy[ylow * ENERGY_SIZE + xhigh] - samEnergy) +
                        10 * (energy[ysam * ENERGY_SIZE + xhigh] - samEnergy) +
                        3 * (energy[yhigh * ENERGY_SIZE + xhigh] - samEnergy);
            xgrad = xgrad / (32 * gridScale);

            f32 ygrad = -3 * (energy[ylow * ENERGY_SIZE + xlow] - samEnergy) - 
                        10 * (energy[ylow * ENERGY_SIZE + xsam] - samEnergy) -
                        3 * (energy[ylow * ENERGY_SIZE + xhigh] - samEnergy) +
                        3 * (energy[yhigh * ENERGY_SIZE + xlow] - samEnergy) +
                        10 * (energy[yhigh * ENERGY_SIZE + xsam] - samEnergy) +
                        3 * (energy[yhigh * ENERGY_SIZE + xhigh] - samEnergy);
            ygrad = ygrad / (32 * gridScale);

            samples->data[i] -= xgrad * stepSize + noiseScale * rand_normal();
            samples->data[i+1] -= ygrad * stepSize + noiseScale * rand_normal();

        }
        clamp_mat_01_(samples);
    }
}

void langevin_samples_ebm(Net* net, Mat* inters, Mat* interGrads, int nSteps, f32 stepSize, f32 energyScale) {
    f32 noiseScale = sqrtf(2 * stepSize);
    for (int step=0; step < nSteps; step++) {
        /// zero grads
        zero_grads_(net, interGrads);
        /// forward
        forward_net(net, inters);
        /// use value of out for out grad
        copy_f32(inters[net->nLayers].data, interGrads[net->nLayers].data, interGrads[net->nLayers].rows);

        mat_const_mul_(&interGrads[net->nLayers], energyScale);

        backward_net(net, inters, interGrads);
        
        // step
        // x = x - a * grad + sqrt(2 * a) * normal
        mat_const_mul_(&interGrads[0], stepSize);
        for (int i=0; i < interGrads[0].rows * interGrads[0].cols; i++) {
            interGrads[0].data[i] += noiseScale * rand_normal();
        }

        sub_mat_(&inters[0], &interGrads[0]);

        clamp_mat_01_(&inters[0]);
    }
}
///--- END EBM ---///

f32 energy_func(f32 x, f32 y, f32 *energyLandscape, int width, int height) {
    // clamp x,y to [0,1]
    x = x * (x > 0) * (x < 1) + 1 * (x >= 1);
    y = y * (y > 0) * (y < 1) + 1 * (y >= 1);
    

    // Get closest low and high cell coords
    int xLow = (int)(x * (width-2));
    int xHigh = xLow + 1;

    int yLow = (int)(y * (height-2));
    int yHigh = yLow + 1;

    // xLow <= x, xHigh > x
    f32 xLowDist = x - xLow;
    f32 xHighDist = xHigh - x;

    f32 yLowDist = y - yLow;
    f32 yHighDist = yHigh - y;

    f32 ll = energyLandscape[width * yLow + xLow];
    f32 lh = energyLandscape[width * yHigh + xLow];
    f32 hl = energyLandscape[width * yHigh + xHigh];
    f32 hh = energyLandscape[width * yLow + xHigh];

    return ll * xLowDist * yLowDist + 
           lh * xLowDist * yHighDist + 
           hl * xHighDist * yLowDist + 
           hh * xHighDist * yHighDist; 
}

///--- BEGIN INTERFACE ---///


EMSCRIPTEN_KEEPALIVE
byte* get_energy_ptr(AppState* state) {
    return (byte*)state->energy;
}


EMSCRIPTEN_KEEPALIVE
byte* get_pixels_ptr(AppState* state) {
    return (byte*)state->pixels;
}


EMSCRIPTEN_KEEPALIVE
byte* get_ebm_pixels_ptr(AppState* state) {
    return (byte*)state->ebmPixels;
}

EMSCRIPTEN_KEEPALIVE
void train_ebm(AppState* state, int steps, f32 lr) {

    Net* net = state->ebmNet;
    Mat* inters = state->ebmInters;
    Mat* interGrads = state->ebmInterGrads;
    Mat* samples = state->ebmSamples;

    Mat* outs = state->ebmTrainOuts;

    for (int step=0; step < steps; step++) {
        int batchStart = rand() % ENERGY_SIZE;
        f32* inpData = inters[0].data;
        f32* outData = outs->data;
        // copy to input
        for (int i=batchStart; i < ENERGY_SIZE * ENERGY_SIZE; i+=ENERGY_SIZE) {
            // calc x and norm to [0,1]
            f32 x = (f32)(i % ENERGY_SIZE) / (f32) ENERGY_SIZE;
            f32 y = (f32)(i / ENERGY_SIZE) / (f32) ENERGY_SIZE;

            *inpData++ = x;
            *inpData++ = y;
            *outData++ = state->energy[i];
        }

        // sample langevin
        //copy_f32(samples->data, inters[0].data, inters[0].rows * inters[0].cols);
        //langevin_samples(net, inters, interGrads, 0.05, 10);
        //copy_f32(inters[0].data, samples->data, inters[0].rows * inters[0].cols);
        

        train_step(net, inters, interGrads, outs, lr);
    }

}


void render_energy_to_pixels(f32* energy, byte* pixels) {
    for (int y=0; y < ENERGY_SIZE; y++) {
        for (int x=0; x < ENERGY_SIZE; x++) {
            f32 val = energy[y * ENERGY_SIZE + x];
            val = (val > 0 && val <= 1) * val + (val > 1);
            // RGBA & LITTLE ENDIAN
            // A
            *pixels++ = 255; 
            // B
            *pixels++ = 0;
            // G
            *pixels++ = (byte) (val * 255);
            // R
            *pixels++ = 255;
        }
    }
}


EMSCRIPTEN_KEEPALIVE
void init_ebm_samples(AppState* state) {
    rand_mat_(state->ebmSamples, 0, 1);
}

EMSCRIPTEN_KEEPALIVE
void sample_ebm(AppState* state, int steps, f32 stepSize, f32 energyScale) {
    Mat* samples = state->ebmSamples;
    Mat* inters = state->ebmInters;
    Mat* interGrads = state->ebmInterGrads;
    Net* net = state->ebmNet;

    copy_f32(samples->data, inters[0].data, inters[0].rows * inters[0].cols);
    langevin_samples_ebm(net, inters, interGrads, steps, stepSize, energyScale);
    copy_f32(inters[0].data, samples->data, inters[0].rows * inters[0].cols);
}


void render_samples_to_pixels(Mat* samples, byte* pixels) {
    f32* sampleData = samples->data;
    for (int i=0; i < samples->rows; i++) {
        int posx = (int) ((*sampleData++) * ENERGY_SIZE);
        int posy = (int) ((*sampleData++) * ENERGY_SIZE);
        posx = (posx >= 0 && posx < ENERGY_SIZE) * posx + (ENERGY_SIZE-1) * (posx >= ENERGY_SIZE);
        posy = (posy >= 0 && posy < ENERGY_SIZE) * posy + (ENERGY_SIZE-1) * (posy >= ENERGY_SIZE);

        int idx = posy * ENERGY_SIZE * 4 + posx * 4;
        pixels[idx] = 255;
        pixels[idx + 1] = 255;
        pixels[idx + 2] = 255;
        pixels[idx + 3] = 255;
    }

}

EMSCRIPTEN_KEEPALIVE
void render_ebm_pixels(AppState* state) {
    Net* net = state->ebmNet;
    Mat* inters = state->ebmInters;
    f32* energy = state->ebmEnergy;
    Mat* samples = state->ebmSamples;

    for (int j=0; j < ENERGY_SIZE; j++) {
        // copy over to input
        f32* inpData = inters[0].data;
        for (int i=0; i < ENERGY_SIZE; i++) {
            // calc x and norm to [0,1]
            f32 x = i / (f32) ENERGY_SIZE;
            f32 y = j / (f32) ENERGY_SIZE;

            *inpData++ = x;
            *inpData++ = y;
        }

        // Run net just on this row
        forward_net(net, inters);

        // copy from output to energy
        copy_f32(inters[net->nLayers].data, &energy[j * ENERGY_SIZE], ENERGY_SIZE);
    }

    render_energy_to_pixels(energy, state->ebmPixels);

    render_samples_to_pixels(state->ebmSamples, state->ebmPixels);
}


EMSCRIPTEN_KEEPALIVE
void init_energy_samples(AppState* state) {
    rand_mat_(state->energySamples, 0, 1);
}

EMSCRIPTEN_KEEPALIVE
void sample_energy(AppState* state, int steps, f32 stepSize, f32 energyScale) {
    langevin_samples(state->energy, state->energySamples, steps, stepSize, energyScale);
}

EMSCRIPTEN_KEEPALIVE
void render_pixels(AppState* state) {
    render_energy_to_pixels(state->energy, state->pixels);
    render_samples_to_pixels(state->energySamples, state->pixels);
}


EMSCRIPTEN_KEEPALIVE
AppState* setup() {
    srand(time(0));
    size_t total = sizeof(AppState) + ARENA_BYTES;    
    AppState* s = (AppState*) malloc(total);
    if (!s) return 0;
    s->arena.base = (byte*)(s + sizeof(AppState));
    s->arena.size = ARENA_BYTES;
    
    s->energy = (f32*) arena_alloc(&s->arena, sizeof(f32) * ENERGY_SIZE * ENERGY_SIZE);
    
    s->pixels = (byte*) arena_alloc(&s->arena, sizeof(byte) * 4 * ENERGY_SIZE * ENERGY_SIZE);
    s->energySamples = alloc_mat(ENERGY_SIZE, 2, &s->arena);
    

    ///--- EBM ---///
    int netDims[] = {2, 32, 64, 32, 1};
    s->ebmNet = alloc_net(5, netDims, &s->arena);
    xavier_init(s->ebmNet);
    s->ebmInters = alloc_inters(ENERGY_SIZE, s->ebmNet, &s->arena);
    s->ebmInterGrads = alloc_inters(ENERGY_SIZE, s->ebmNet, &s->arena);

    s->ebmSamples = alloc_mat(ENERGY_SIZE, 2, &s->arena);
    s->ebmTrainOuts = alloc_mat(ENERGY_SIZE, 1, &s->arena);
    s->ebmEnergy = (f32*) arena_alloc(&s->arena, sizeof(f32) * ENERGY_SIZE * ENERGY_SIZE);
    s->ebmPixels = (byte*) arena_alloc(&s->arena, sizeof(byte) * 4 * ENERGY_SIZE * ENERGY_SIZE);
    ///--- END EBM ---///

    return s;
}

///--- END INTERFACE ---///
