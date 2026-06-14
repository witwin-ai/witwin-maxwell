#ifndef MAXWELL_FDTD_ADJOINT_KERNELS_SLANG
#define MAXWELL_FDTD_ADJOINT_KERNELS_SLANG

#include "../boundary/helpers.slang"

uint linearThreadIndex1D()
{
    return cudaThreadIdx().x + cudaBlockIdx().x * cudaBlockDim().x;
}

uint linearSize3D(TensorView<float> field)
{
    return field.size(0) * field.size(1) * field.size(2);
}

uint3 unflatten3D(uint linearIndex, uint sizeY, uint sizeZ)
{
    uint planeSize = sizeY * sizeZ;
    uint i = linearIndex / planeSize;
    uint remainder = linearIndex - i * planeSize;
    uint j = remainder / sizeZ;
    uint k = remainder - j * sizeZ;
    return uint3(i, j, k);
}

bool isValidIndex3D(int i, int j, int k, TensorView<float> field)
{
    return i >= 0
        && i < int(field.size(0))
        && j >= 0
        && j < int(field.size(1))
        && k >= 0
        && k < int(field.size(2));
}

bool isInteriorCoordinate(uint coordinate, uint size)
{
    return coordinate > 0 && coordinate + 1 < size;
}

bool isExActiveIndex(int i, int j, int k, TensorView<float> Ex)
{
    if (!isValidIndex3D(i, j, k, Ex))
    {
        return false;
    }
    return isInteriorCoordinate(uint(j), Ex.size(1)) && isInteriorCoordinate(uint(k), Ex.size(2));
}

bool isEyActiveIndex(int i, int j, int k, TensorView<float> Ey)
{
    if (!isValidIndex3D(i, j, k, Ey))
    {
        return false;
    }
    return isInteriorCoordinate(uint(i), Ey.size(0)) && isInteriorCoordinate(uint(k), Ey.size(2));
}

bool isEzActiveIndex(int i, int j, int k, TensorView<float> Ez)
{
    if (!isValidIndex3D(i, j, k, Ez))
    {
        return false;
    }
    return isInteriorCoordinate(uint(i), Ez.size(0)) && isInteriorCoordinate(uint(j), Ez.size(1));
}

struct ElectricCellStatus
{
    bool active;
    bool inactive;
    bool pec;
};

ElectricCellStatus resolveElectricCellStatus(
    uint coordinateA,
    uint sizeA,
    uint lowModeA,
    uint highModeA,
    uint coordinateB,
    uint sizeB,
    uint lowModeB,
    uint highModeB
)
{
    ElectricCellStatus status;

    bool axisAPec = false;
    bool axisAInactive = false;
    if (isBoundaryIndex(coordinateA, sizeA))
    {
        uint modeA = selectBoundaryMode(lowModeA, highModeA, coordinateA);
        axisAPec = isPecBoundaryMode(modeA);
        axisAInactive = isInactiveBoundaryMode(modeA);
    }

    bool axisBPec = false;
    bool axisBInactive = false;
    if (isBoundaryIndex(coordinateB, sizeB))
    {
        uint modeB = selectBoundaryMode(lowModeB, highModeB, coordinateB);
        axisBPec = isPecBoundaryMode(modeB);
        axisBInactive = isInactiveBoundaryMode(modeB);
    }

    status.pec = axisAPec || axisBPec;
    status.inactive = !status.pec && (axisAInactive || axisBInactive);
    status.active = !status.pec && !status.inactive;
    return status;
}

float2 loadComplex3D(TensorView<float> realField, TensorView<float> imagField, uint i, uint j, uint k)
{
    return float2(realField[i, j, k], imagField[i, j, k]);
}

float2 loadScaledAdjoint(
    TensorView<float> adjReal,
    TensorView<float> adjImag,
    TensorView<float> curl,
    uint i,
    uint j,
    uint k,
    float sign
)
{
    float scale = sign * curl[i, j, k];
    return float2(adjReal[i, j, k] * scale, adjImag[i, j, k] * scale);
}

float complexInnerReal(float2 lhs, float2 rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y;
}

float2 blochBackwardDiffX(
    TensorView<float> realField,
    TensorView<float> imagField,
    uint i,
    uint j,
    uint k,
    float phaseCos,
    float phaseSin,
    float invDx
)
{
    uint fieldSize = realField.size(0);
    if (i == 0 || i == fieldSize)
    {
        DiffFloat2 diff = blochBackwardDiff(
            i,
            loadComplex3D(realField, imagField, 0, j, k),
            loadComplex3D(realField, imagField, realField.size(0) - 1, j, k),
            phaseCos,
            phaseSin,
            invDx
        );
        return diff.value;
    }

    return (loadComplex3D(realField, imagField, i, j, k) - loadComplex3D(realField, imagField, i - 1, j, k)) * invDx;
}

float2 blochBackwardDiffY(
    TensorView<float> realField,
    TensorView<float> imagField,
    uint i,
    uint j,
    uint k,
    float phaseCos,
    float phaseSin,
    float invDy
)
{
    uint fieldSize = realField.size(1);
    if (j == 0 || j == fieldSize)
    {
        DiffFloat2 diff = blochBackwardDiff(
            j,
            loadComplex3D(realField, imagField, i, 0, k),
            loadComplex3D(realField, imagField, i, realField.size(1) - 1, k),
            phaseCos,
            phaseSin,
            invDy
        );
        return diff.value;
    }

    return (loadComplex3D(realField, imagField, i, j, k) - loadComplex3D(realField, imagField, i, j - 1, k)) * invDy;
}

float2 blochBackwardDiffZ(
    TensorView<float> realField,
    TensorView<float> imagField,
    uint i,
    uint j,
    uint k,
    float phaseCos,
    float phaseSin,
    float invDz
)
{
    uint fieldSize = realField.size(2);
    if (k == 0 || k == fieldSize)
    {
        DiffFloat2 diff = blochBackwardDiff(
            k,
            loadComplex3D(realField, imagField, i, j, 0),
            loadComplex3D(realField, imagField, i, j, realField.size(2) - 1),
            phaseCos,
            phaseSin,
            invDz
        );
        return diff.value;
    }

    return (loadComplex3D(realField, imagField, i, j, k) - loadComplex3D(realField, imagField, i, j, k - 1)) * invDz;
}

float2 gatherBlochBackwardDiffAdjointX(
    TensorView<float> adjReal,
    TensorView<float> adjImag,
    TensorView<float> curl,
    uint i,
    uint j,
    uint k,
    float sign,
    float phaseCos,
    float phaseSin,
    float invDx
)
{
    uint fieldSize = adjReal.size(0) - 1;
    float2 value = float2(0.0, 0.0);

    if (i > 0)
    {
        value += invDx * loadScaledAdjoint(adjReal, adjImag, curl, i, j, k, sign);
    }
    if (i + 1 < fieldSize)
    {
        value -= invDx * loadScaledAdjoint(adjReal, adjImag, curl, i + 1, j, k, sign);
    }

    float2 low = loadScaledAdjoint(adjReal, adjImag, curl, 0, j, k, sign);
    float2 high = loadScaledAdjoint(adjReal, adjImag, curl, adjReal.size(0) - 1, j, k, sign);
    if (i == 0)
    {
        value += invDx * low + invDx * complexPhaseNegative(phaseCos, phaseSin, high);
    }
    if (i + 1 == fieldSize)
    {
        value += -invDx * complexPhasePositive(phaseCos, phaseSin, low) - invDx * high;
    }

    return value;
}

float2 gatherBlochBackwardDiffAdjointY(
    TensorView<float> adjReal,
    TensorView<float> adjImag,
    TensorView<float> curl,
    uint i,
    uint j,
    uint k,
    float sign,
    float phaseCos,
    float phaseSin,
    float invDy
)
{
    uint fieldSize = adjReal.size(1) - 1;
    float2 value = float2(0.0, 0.0);

    if (j > 0)
    {
        value += invDy * loadScaledAdjoint(adjReal, adjImag, curl, i, j, k, sign);
    }
    if (j + 1 < fieldSize)
    {
        value -= invDy * loadScaledAdjoint(adjReal, adjImag, curl, i, j + 1, k, sign);
    }

    float2 low = loadScaledAdjoint(adjReal, adjImag, curl, i, 0, k, sign);
    float2 high = loadScaledAdjoint(adjReal, adjImag, curl, i, adjReal.size(1) - 1, k, sign);
    if (j == 0)
    {
        value += invDy * low + invDy * complexPhaseNegative(phaseCos, phaseSin, high);
    }
    if (j + 1 == fieldSize)
    {
        value += -invDy * complexPhasePositive(phaseCos, phaseSin, low) - invDy * high;
    }

    return value;
}

float2 gatherBlochBackwardDiffAdjointZ(
    TensorView<float> adjReal,
    TensorView<float> adjImag,
    TensorView<float> curl,
    uint i,
    uint j,
    uint k,
    float sign,
    float phaseCos,
    float phaseSin,
    float invDz
)
{
    uint fieldSize = adjReal.size(2) - 1;
    float2 value = float2(0.0, 0.0);

    if (k > 0)
    {
        value += invDz * loadScaledAdjoint(adjReal, adjImag, curl, i, j, k, sign);
    }
    if (k + 1 < fieldSize)
    {
        value -= invDz * loadScaledAdjoint(adjReal, adjImag, curl, i, j, k + 1, sign);
    }

    float2 low = loadScaledAdjoint(adjReal, adjImag, curl, i, j, 0, sign);
    float2 high = loadScaledAdjoint(adjReal, adjImag, curl, i, j, adjReal.size(2) - 1, sign);
    if (k == 0)
    {
        value += invDz * low + invDz * complexPhaseNegative(phaseCos, phaseSin, high);
    }
    if (k + 1 == fieldSize)
    {
        value += -invDz * complexPhasePositive(phaseCos, phaseSin, low) - invDz * high;
    }

    return value;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseElectricAdjointToHxBloch3D(
    TensorView<float> AdjHxMidReal,
    TensorView<float> AdjHxMidImag,
    TensorView<float> AdjHxPostReal,
    TensorView<float> AdjHxPostImag,
    TensorView<float> AdjEyPostReal,
    TensorView<float> AdjEyPostImag,
    TensorView<float> AdjEzPostReal,
    TensorView<float> AdjEzPostImag,
    TensorView<float> EyCurl,
    TensorView<float> EzCurl,
    float phaseCosY,
    float phaseSinY,
    float phaseCosZ,
    float phaseSinZ,
    float invDy,
    float invDz
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjHxMidReal);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjHxMidReal.size(1), AdjHxMidReal.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float2 adjoint = loadComplex3D(AdjHxPostReal, AdjHxPostImag, i, j, k);
    adjoint += gatherBlochBackwardDiffAdjointZ(
        AdjEyPostReal,
        AdjEyPostImag,
        EyCurl,
        i,
        j,
        k,
        1.0f,
        phaseCosZ,
        phaseSinZ,
        invDz
    );
    adjoint += gatherBlochBackwardDiffAdjointY(
        AdjEzPostReal,
        AdjEzPostImag,
        EzCurl,
        i,
        j,
        k,
        -1.0f,
        phaseCosY,
        phaseSinY,
        invDy
    );

    AdjHxMidReal[i, j, k] = adjoint.x;
    AdjHxMidImag[i, j, k] = adjoint.y;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseElectricAdjointToHyBloch3D(
    TensorView<float> AdjHyMidReal,
    TensorView<float> AdjHyMidImag,
    TensorView<float> AdjHyPostReal,
    TensorView<float> AdjHyPostImag,
    TensorView<float> AdjExPostReal,
    TensorView<float> AdjExPostImag,
    TensorView<float> AdjEzPostReal,
    TensorView<float> AdjEzPostImag,
    TensorView<float> ExCurl,
    TensorView<float> EzCurl,
    float phaseCosX,
    float phaseSinX,
    float phaseCosZ,
    float phaseSinZ,
    float invDx,
    float invDz
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjHyMidReal);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjHyMidReal.size(1), AdjHyMidReal.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float2 adjoint = loadComplex3D(AdjHyPostReal, AdjHyPostImag, i, j, k);
    adjoint += gatherBlochBackwardDiffAdjointZ(
        AdjExPostReal,
        AdjExPostImag,
        ExCurl,
        i,
        j,
        k,
        -1.0f,
        phaseCosZ,
        phaseSinZ,
        invDz
    );
    adjoint += gatherBlochBackwardDiffAdjointX(
        AdjEzPostReal,
        AdjEzPostImag,
        EzCurl,
        i,
        j,
        k,
        1.0f,
        phaseCosX,
        phaseSinX,
        invDx
    );

    AdjHyMidReal[i, j, k] = adjoint.x;
    AdjHyMidImag[i, j, k] = adjoint.y;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseElectricAdjointToHzBloch3D(
    TensorView<float> AdjHzMidReal,
    TensorView<float> AdjHzMidImag,
    TensorView<float> AdjHzPostReal,
    TensorView<float> AdjHzPostImag,
    TensorView<float> AdjExPostReal,
    TensorView<float> AdjExPostImag,
    TensorView<float> AdjEyPostReal,
    TensorView<float> AdjEyPostImag,
    TensorView<float> ExCurl,
    TensorView<float> EyCurl,
    float phaseCosX,
    float phaseSinX,
    float phaseCosY,
    float phaseSinY,
    float invDx,
    float invDy
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjHzMidReal);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjHzMidReal.size(1), AdjHzMidReal.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float2 adjoint = loadComplex3D(AdjHzPostReal, AdjHzPostImag, i, j, k);
    adjoint += gatherBlochBackwardDiffAdjointY(
        AdjExPostReal,
        AdjExPostImag,
        ExCurl,
        i,
        j,
        k,
        1.0f,
        phaseCosY,
        phaseSinY,
        invDy
    );
    adjoint += gatherBlochBackwardDiffAdjointX(
        AdjEyPostReal,
        AdjEyPostImag,
        EyCurl,
        i,
        j,
        k,
        -1.0f,
        phaseCosX,
        phaseSinX,
        invDx
    );

    AdjHzMidReal[i, j, k] = adjoint.x;
    AdjHzMidImag[i, j, k] = adjoint.y;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseMagneticAdjointToExBloch3D(
    TensorView<float> AdjExPrevReal,
    TensorView<float> AdjExPrevImag,
    TensorView<float> GradEpsEx,
    TensorView<float> AdjExPostReal,
    TensorView<float> AdjExPostImag,
    TensorView<float> AdjHyMidReal,
    TensorView<float> AdjHyMidImag,
    TensorView<float> AdjHzMidReal,
    TensorView<float> AdjHzMidImag,
    TensorView<float> ExDecay,
    TensorView<float> ExCurl,
    TensorView<float> EpsEx,
    TensorView<float> HyMidReal,
    TensorView<float> HyMidImag,
    TensorView<float> HzMidReal,
    TensorView<float> HzMidImag,
    TensorView<float> HyCurl,
    TensorView<float> HzCurl,
    float phaseCosY,
    float phaseSinY,
    float phaseCosZ,
    float phaseSinZ,
    float invDy,
    float invDz
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjExPrevReal);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjExPrevReal.size(1), AdjExPrevReal.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float2 adjExPost = loadComplex3D(AdjExPostReal, AdjExPostImag, i, j, k);
    float2 dHzDy = blochBackwardDiffY(HzMidReal, HzMidImag, i, j, k, phaseCosY, phaseSinY, invDy);
    float2 dHyDz = blochBackwardDiffZ(HyMidReal, HyMidImag, i, j, k, phaseCosZ, phaseSinZ, invDz);
    float2 curlH = dHzDy - dHyDz;

    float adjointReal = adjExPost.x * ExDecay[i, j, k];
    float adjointImag = adjExPost.y * ExDecay[i, j, k];
    float grad = -ExCurl[i, j, k] * complexInnerReal(adjExPost, curlH) / EpsEx[i, j, k];

    if (k < AdjHyMidReal.size(2))
    {
        adjointReal += HyCurl[i, j, k] * invDz * AdjHyMidReal[i, j, k];
        adjointImag += HyCurl[i, j, k] * invDz * AdjHyMidImag[i, j, k];
    }
    if (k > 0)
    {
        uint prevK = k - 1;
        adjointReal -= HyCurl[i, j, prevK] * invDz * AdjHyMidReal[i, j, prevK];
        adjointImag -= HyCurl[i, j, prevK] * invDz * AdjHyMidImag[i, j, prevK];
    }
    if (j < AdjHzMidReal.size(1))
    {
        adjointReal -= HzCurl[i, j, k] * invDy * AdjHzMidReal[i, j, k];
        adjointImag -= HzCurl[i, j, k] * invDy * AdjHzMidImag[i, j, k];
    }
    if (j > 0)
    {
        uint prevJ = j - 1;
        adjointReal += HzCurl[i, prevJ, k] * invDy * AdjHzMidReal[i, prevJ, k];
        adjointImag += HzCurl[i, prevJ, k] * invDy * AdjHzMidImag[i, prevJ, k];
    }

    AdjExPrevReal[i, j, k] = adjointReal;
    AdjExPrevImag[i, j, k] = adjointImag;
    GradEpsEx[i, j, k] = grad;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseMagneticAdjointToEyBloch3D(
    TensorView<float> AdjEyPrevReal,
    TensorView<float> AdjEyPrevImag,
    TensorView<float> GradEpsEy,
    TensorView<float> AdjEyPostReal,
    TensorView<float> AdjEyPostImag,
    TensorView<float> AdjHxMidReal,
    TensorView<float> AdjHxMidImag,
    TensorView<float> AdjHzMidReal,
    TensorView<float> AdjHzMidImag,
    TensorView<float> EyDecay,
    TensorView<float> EyCurl,
    TensorView<float> EpsEy,
    TensorView<float> HxMidReal,
    TensorView<float> HxMidImag,
    TensorView<float> HzMidReal,
    TensorView<float> HzMidImag,
    TensorView<float> HxCurl,
    TensorView<float> HzCurl,
    float phaseCosX,
    float phaseSinX,
    float phaseCosZ,
    float phaseSinZ,
    float invDx,
    float invDz
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjEyPrevReal);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjEyPrevReal.size(1), AdjEyPrevReal.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float2 adjEyPost = loadComplex3D(AdjEyPostReal, AdjEyPostImag, i, j, k);
    float2 dHxDz = blochBackwardDiffZ(HxMidReal, HxMidImag, i, j, k, phaseCosZ, phaseSinZ, invDz);
    float2 dHzDx = blochBackwardDiffX(HzMidReal, HzMidImag, i, j, k, phaseCosX, phaseSinX, invDx);
    float2 curlH = dHxDz - dHzDx;

    float adjointReal = adjEyPost.x * EyDecay[i, j, k];
    float adjointImag = adjEyPost.y * EyDecay[i, j, k];
    float grad = -EyCurl[i, j, k] * complexInnerReal(adjEyPost, curlH) / EpsEy[i, j, k];

    if (k < AdjHxMidReal.size(2))
    {
        adjointReal -= HxCurl[i, j, k] * invDz * AdjHxMidReal[i, j, k];
        adjointImag -= HxCurl[i, j, k] * invDz * AdjHxMidImag[i, j, k];
    }
    if (k > 0)
    {
        uint prevK = k - 1;
        adjointReal += HxCurl[i, j, prevK] * invDz * AdjHxMidReal[i, j, prevK];
        adjointImag += HxCurl[i, j, prevK] * invDz * AdjHxMidImag[i, j, prevK];
    }
    if (i < AdjHzMidReal.size(0))
    {
        adjointReal += HzCurl[i, j, k] * invDx * AdjHzMidReal[i, j, k];
        adjointImag += HzCurl[i, j, k] * invDx * AdjHzMidImag[i, j, k];
    }
    if (i > 0)
    {
        uint prevI = i - 1;
        adjointReal -= HzCurl[prevI, j, k] * invDx * AdjHzMidReal[prevI, j, k];
        adjointImag -= HzCurl[prevI, j, k] * invDx * AdjHzMidImag[prevI, j, k];
    }

    AdjEyPrevReal[i, j, k] = adjointReal;
    AdjEyPrevImag[i, j, k] = adjointImag;
    GradEpsEy[i, j, k] = grad;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseMagneticAdjointToEzBloch3D(
    TensorView<float> AdjEzPrevReal,
    TensorView<float> AdjEzPrevImag,
    TensorView<float> GradEpsEz,
    TensorView<float> AdjEzPostReal,
    TensorView<float> AdjEzPostImag,
    TensorView<float> AdjHxMidReal,
    TensorView<float> AdjHxMidImag,
    TensorView<float> AdjHyMidReal,
    TensorView<float> AdjHyMidImag,
    TensorView<float> EzDecay,
    TensorView<float> EzCurl,
    TensorView<float> EpsEz,
    TensorView<float> HxMidReal,
    TensorView<float> HxMidImag,
    TensorView<float> HyMidReal,
    TensorView<float> HyMidImag,
    TensorView<float> HxCurl,
    TensorView<float> HyCurl,
    float phaseCosX,
    float phaseSinX,
    float phaseCosY,
    float phaseSinY,
    float invDx,
    float invDy
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjEzPrevReal);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjEzPrevReal.size(1), AdjEzPrevReal.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float2 adjEzPost = loadComplex3D(AdjEzPostReal, AdjEzPostImag, i, j, k);
    float2 dHyDx = blochBackwardDiffX(HyMidReal, HyMidImag, i, j, k, phaseCosX, phaseSinX, invDx);
    float2 dHxDy = blochBackwardDiffY(HxMidReal, HxMidImag, i, j, k, phaseCosY, phaseSinY, invDy);
    float2 curlH = dHyDx - dHxDy;

    float adjointReal = adjEzPost.x * EzDecay[i, j, k];
    float adjointImag = adjEzPost.y * EzDecay[i, j, k];
    float grad = -EzCurl[i, j, k] * complexInnerReal(adjEzPost, curlH) / EpsEz[i, j, k];

    if (j < AdjHxMidReal.size(1))
    {
        adjointReal += HxCurl[i, j, k] * invDy * AdjHxMidReal[i, j, k];
        adjointImag += HxCurl[i, j, k] * invDy * AdjHxMidImag[i, j, k];
    }
    if (j > 0)
    {
        uint prevJ = j - 1;
        adjointReal -= HxCurl[i, prevJ, k] * invDy * AdjHxMidReal[i, prevJ, k];
        adjointImag -= HxCurl[i, prevJ, k] * invDy * AdjHxMidImag[i, prevJ, k];
    }
    if (i < AdjHyMidReal.size(0))
    {
        adjointReal -= HyCurl[i, j, k] * invDx * AdjHyMidReal[i, j, k];
        adjointImag -= HyCurl[i, j, k] * invDx * AdjHyMidImag[i, j, k];
    }
    if (i > 0)
    {
        uint prevI = i - 1;
        adjointReal += HyCurl[prevI, j, k] * invDx * AdjHyMidReal[prevI, j, k];
        adjointImag += HyCurl[prevI, j, k] * invDx * AdjHyMidImag[prevI, j, k];
    }

    AdjEzPrevReal[i, j, k] = adjointReal;
    AdjEzPrevImag[i, j, k] = adjointImag;
    GradEpsEz[i, j, k] = grad;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseDebyeCurrent3D(
    TensorView<float> AdjElectricPrev,
    TensorView<float> AdjPolarizationPrev,
    TensorView<float> AdjPolarizationPost,
    TensorView<float> AdjCurrentPost,
    TensorView<float> DebyeDrive,
    float decay,
    float dt
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjElectricPrev);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjElectricPrev.size(1), AdjElectricPrev.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float adjPolarizationInternal = AdjPolarizationPost[i, j, k] + AdjCurrentPost[i, j, k] / dt;
    AdjElectricPrev[i, j, k] += DebyeDrive[i, j, k] * adjPolarizationInternal;
    AdjPolarizationPrev[i, j, k] += decay * adjPolarizationInternal - AdjCurrentPost[i, j, k] / dt;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseDrudeCurrent3D(
    TensorView<float> AdjElectricPrev,
    TensorView<float> AdjCurrentPrev,
    TensorView<float> AdjCurrentPost,
    TensorView<float> DrudeDrive,
    float decay
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjElectricPrev);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjElectricPrev.size(1), AdjElectricPrev.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float adjCurrent = AdjCurrentPost[i, j, k];
    AdjElectricPrev[i, j, k] += DrudeDrive[i, j, k] * adjCurrent;
    AdjCurrentPrev[i, j, k] += decay * adjCurrent;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseLorentzCurrent3D(
    TensorView<float> AdjElectricPrev,
    TensorView<float> AdjPolarizationPrev,
    TensorView<float> AdjCurrentPrev,
    TensorView<float> AdjPolarizationPost,
    TensorView<float> AdjCurrentPost,
    TensorView<float> LorentzDrive,
    float decay,
    float restoring,
    float dt
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjElectricPrev);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjElectricPrev.size(1), AdjElectricPrev.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float adjCurrentInternal = AdjCurrentPost[i, j, k] + dt * AdjPolarizationPost[i, j, k];
    AdjElectricPrev[i, j, k] += LorentzDrive[i, j, k] * adjCurrentInternal;
    AdjPolarizationPrev[i, j, k] += AdjPolarizationPost[i, j, k] - restoring * adjCurrentInternal;
    AdjCurrentPrev[i, j, k] += decay * adjCurrentInternal;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseElectricAdjointToHxStandard3D(
    TensorView<float> AdjHxMid,
    TensorView<float> AdjHxPost,
    TensorView<float> AdjEyPost,
    TensorView<float> AdjEzPost,
    TensorView<float> EyCurl,
    TensorView<float> EzCurl,
    float invDy,
    float invDz
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjHxMid);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjHxMid.size(1), AdjHxMid.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;
    int ii = int(i);
    int jj = int(j);
    int kk = int(k);

    float adjoint = AdjHxPost[i, j, k];

    if (isEyActiveIndex(ii, jj, kk, AdjEyPost))
    {
        adjoint += EyCurl[i, j, k] * invDz * AdjEyPost[i, j, k];
    }
    if (isEyActiveIndex(ii, jj, kk + 1, AdjEyPost))
    {
        uint nextK = uint(kk + 1);
        adjoint -= EyCurl[i, j, nextK] * invDz * AdjEyPost[i, j, nextK];
    }
    if (isEzActiveIndex(ii, jj, kk, AdjEzPost))
    {
        adjoint -= EzCurl[i, j, k] * invDy * AdjEzPost[i, j, k];
    }
    if (isEzActiveIndex(ii, jj + 1, kk, AdjEzPost))
    {
        uint nextJ = uint(jj + 1);
        adjoint += EzCurl[i, nextJ, k] * invDy * AdjEzPost[i, nextJ, k];
    }

    AdjHxMid[i, j, k] = adjoint;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseElectricAdjointToHyStandard3D(
    TensorView<float> AdjHyMid,
    TensorView<float> AdjHyPost,
    TensorView<float> AdjExPost,
    TensorView<float> AdjEzPost,
    TensorView<float> ExCurl,
    TensorView<float> EzCurl,
    float invDx,
    float invDz
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjHyMid);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjHyMid.size(1), AdjHyMid.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;
    int ii = int(i);
    int jj = int(j);
    int kk = int(k);

    float adjoint = AdjHyPost[i, j, k];

    if (isExActiveIndex(ii, jj, kk, AdjExPost))
    {
        adjoint -= ExCurl[i, j, k] * invDz * AdjExPost[i, j, k];
    }
    if (isExActiveIndex(ii, jj, kk + 1, AdjExPost))
    {
        uint nextK = uint(kk + 1);
        adjoint += ExCurl[i, j, nextK] * invDz * AdjExPost[i, j, nextK];
    }
    if (isEzActiveIndex(ii, jj, kk, AdjEzPost))
    {
        adjoint += EzCurl[i, j, k] * invDx * AdjEzPost[i, j, k];
    }
    if (isEzActiveIndex(ii + 1, jj, kk, AdjEzPost))
    {
        uint nextI = uint(ii + 1);
        adjoint -= EzCurl[nextI, j, k] * invDx * AdjEzPost[nextI, j, k];
    }

    AdjHyMid[i, j, k] = adjoint;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseElectricAdjointToHzStandard3D(
    TensorView<float> AdjHzMid,
    TensorView<float> AdjHzPost,
    TensorView<float> AdjExPost,
    TensorView<float> AdjEyPost,
    TensorView<float> ExCurl,
    TensorView<float> EyCurl,
    float invDx,
    float invDy
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjHzMid);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjHzMid.size(1), AdjHzMid.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;
    int ii = int(i);
    int jj = int(j);
    int kk = int(k);

    float adjoint = AdjHzPost[i, j, k];

    if (isExActiveIndex(ii, jj, kk, AdjExPost))
    {
        adjoint += ExCurl[i, j, k] * invDy * AdjExPost[i, j, k];
    }
    if (isExActiveIndex(ii, jj + 1, kk, AdjExPost))
    {
        uint nextJ = uint(jj + 1);
        adjoint -= ExCurl[i, nextJ, k] * invDy * AdjExPost[i, nextJ, k];
    }
    if (isEyActiveIndex(ii, jj, kk, AdjEyPost))
    {
        adjoint -= EyCurl[i, j, k] * invDx * AdjEyPost[i, j, k];
    }
    if (isEyActiveIndex(ii + 1, jj, kk, AdjEyPost))
    {
        uint nextI = uint(ii + 1);
        adjoint += EyCurl[nextI, j, k] * invDx * AdjEyPost[nextI, j, k];
    }

    AdjHzMid[i, j, k] = adjoint;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseMagneticAdjointToExStandard3D(
    TensorView<float> AdjExPrev,
    TensorView<float> GradEpsEx,
    TensorView<float> AdjExPost,
    TensorView<float> AdjHyMid,
    TensorView<float> AdjHzMid,
    TensorView<float> ExDecay,
    TensorView<float> ExCurl,
    TensorView<float> EpsEx,
    TensorView<float> HyMid,
    TensorView<float> HzMid,
    TensorView<float> HyCurl,
    TensorView<float> HzCurl,
    float invDy,
    float invDz,
    uint yLowBoundaryMode,
    uint yHighBoundaryMode,
    uint zLowBoundaryMode,
    uint zHighBoundaryMode
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjExPrev);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjExPrev.size(1), AdjExPrev.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    ElectricCellStatus status = resolveElectricCellStatus(
        j,
        AdjExPrev.size(1),
        yLowBoundaryMode,
        yHighBoundaryMode,
        k,
        AdjExPrev.size(2),
        zLowBoundaryMode,
        zHighBoundaryMode
    );

    float adjoint = 0.0f;
    float grad = 0.0f;

    if (status.inactive)
    {
        adjoint = AdjExPost[i, j, k];
    }
    else if (status.active)
    {
        uint prevJ = j - 1;
        uint prevK = k - 1;
        float curlH = (HzMid[i, j, k] - HzMid[i, prevJ, k]) * invDy
            - (HyMid[i, j, k] - HyMid[i, j, prevK]) * invDz;
        adjoint = AdjExPost[i, j, k] * ExDecay[i, j, k];
        grad = -AdjExPost[i, j, k] * ExCurl[i, j, k] * curlH / EpsEx[i, j, k];
    }

    if (k < AdjHyMid.size(2))
    {
        adjoint += HyCurl[i, j, k] * invDz * AdjHyMid[i, j, k];
    }
    if (k > 0)
    {
        uint prevK = k - 1;
        adjoint -= HyCurl[i, j, prevK] * invDz * AdjHyMid[i, j, prevK];
    }
    if (j < AdjHzMid.size(1))
    {
        adjoint -= HzCurl[i, j, k] * invDy * AdjHzMid[i, j, k];
    }
    if (j > 0)
    {
        uint prevJ = j - 1;
        adjoint += HzCurl[i, prevJ, k] * invDy * AdjHzMid[i, prevJ, k];
    }

    AdjExPrev[i, j, k] = adjoint;
    GradEpsEx[i, j, k] = grad;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseMagneticAdjointToEyStandard3D(
    TensorView<float> AdjEyPrev,
    TensorView<float> GradEpsEy,
    TensorView<float> AdjEyPost,
    TensorView<float> AdjHxMid,
    TensorView<float> AdjHzMid,
    TensorView<float> EyDecay,
    TensorView<float> EyCurl,
    TensorView<float> EpsEy,
    TensorView<float> HxMid,
    TensorView<float> HzMid,
    TensorView<float> HxCurl,
    TensorView<float> HzCurl,
    float invDx,
    float invDz,
    uint xLowBoundaryMode,
    uint xHighBoundaryMode,
    uint zLowBoundaryMode,
    uint zHighBoundaryMode
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjEyPrev);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjEyPrev.size(1), AdjEyPrev.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    ElectricCellStatus status = resolveElectricCellStatus(
        i,
        AdjEyPrev.size(0),
        xLowBoundaryMode,
        xHighBoundaryMode,
        k,
        AdjEyPrev.size(2),
        zLowBoundaryMode,
        zHighBoundaryMode
    );

    float adjoint = 0.0f;
    float grad = 0.0f;

    if (status.inactive)
    {
        adjoint = AdjEyPost[i, j, k];
    }
    else if (status.active)
    {
        uint prevI = i - 1;
        uint prevK = k - 1;
        float curlH = (HxMid[i, j, k] - HxMid[i, j, prevK]) * invDz
            - (HzMid[i, j, k] - HzMid[prevI, j, k]) * invDx;
        adjoint = AdjEyPost[i, j, k] * EyDecay[i, j, k];
        grad = -AdjEyPost[i, j, k] * EyCurl[i, j, k] * curlH / EpsEy[i, j, k];
    }

    if (k < AdjHxMid.size(2))
    {
        adjoint -= HxCurl[i, j, k] * invDz * AdjHxMid[i, j, k];
    }
    if (k > 0)
    {
        uint prevK = k - 1;
        adjoint += HxCurl[i, j, prevK] * invDz * AdjHxMid[i, j, prevK];
    }
    if (i < AdjHzMid.size(0))
    {
        adjoint += HzCurl[i, j, k] * invDx * AdjHzMid[i, j, k];
    }
    if (i > 0)
    {
        uint prevI = i - 1;
        adjoint -= HzCurl[prevI, j, k] * invDx * AdjHzMid[prevI, j, k];
    }

    AdjEyPrev[i, j, k] = adjoint;
    GradEpsEy[i, j, k] = grad;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseMagneticAdjointToEzStandard3D(
    TensorView<float> AdjEzPrev,
    TensorView<float> GradEpsEz,
    TensorView<float> AdjEzPost,
    TensorView<float> AdjHxMid,
    TensorView<float> AdjHyMid,
    TensorView<float> EzDecay,
    TensorView<float> EzCurl,
    TensorView<float> EpsEz,
    TensorView<float> HxMid,
    TensorView<float> HyMid,
    TensorView<float> HxCurl,
    TensorView<float> HyCurl,
    float invDx,
    float invDy,
    uint xLowBoundaryMode,
    uint xHighBoundaryMode,
    uint yLowBoundaryMode,
    uint yHighBoundaryMode
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjEzPrev);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjEzPrev.size(1), AdjEzPrev.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    ElectricCellStatus status = resolveElectricCellStatus(
        i,
        AdjEzPrev.size(0),
        xLowBoundaryMode,
        xHighBoundaryMode,
        j,
        AdjEzPrev.size(1),
        yLowBoundaryMode,
        yHighBoundaryMode
    );

    float adjoint = 0.0f;
    float grad = 0.0f;

    if (status.inactive)
    {
        adjoint = AdjEzPost[i, j, k];
    }
    else if (status.active)
    {
        uint prevI = i - 1;
        uint prevJ = j - 1;
        float curlH = (HyMid[i, j, k] - HyMid[prevI, j, k]) * invDx
            - (HxMid[i, j, k] - HxMid[i, prevJ, k]) * invDy;
        adjoint = AdjEzPost[i, j, k] * EzDecay[i, j, k];
        grad = -AdjEzPost[i, j, k] * EzCurl[i, j, k] * curlH / EpsEz[i, j, k];
    }

    if (j < AdjHxMid.size(1))
    {
        adjoint += HxCurl[i, j, k] * invDy * AdjHxMid[i, j, k];
    }
    if (j > 0)
    {
        uint prevJ = j - 1;
        adjoint -= HxCurl[i, prevJ, k] * invDy * AdjHxMid[i, prevJ, k];
    }
    if (i < AdjHyMid.size(0))
    {
        adjoint -= HyCurl[i, j, k] * invDx * AdjHyMid[i, j, k];
    }
    if (i > 0)
    {
        uint prevI = i - 1;
        adjoint += HyCurl[prevI, j, k] * invDx * AdjHyMid[prevI, j, k];
    }

    AdjEzPrev[i, j, k] = adjoint;
    GradEpsEz[i, j, k] = grad;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseMagneticAdjointToHxStandard3D(
    TensorView<float> AdjHxPrev,
    TensorView<float> AdjHxMid,
    TensorView<float> HxDecay
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjHxPrev);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjHxPrev.size(1), AdjHxPrev.size(2));
    AdjHxPrev[index.x, index.y, index.z] = AdjHxMid[index.x, index.y, index.z] * HxDecay[index.x, index.y, index.z];
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseMagneticAdjointToHyStandard3D(
    TensorView<float> AdjHyPrev,
    TensorView<float> AdjHyMid,
    TensorView<float> HyDecay
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjHyPrev);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjHyPrev.size(1), AdjHyPrev.size(2));
    AdjHyPrev[index.x, index.y, index.z] = AdjHyMid[index.x, index.y, index.z] * HyDecay[index.x, index.y, index.z];
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseMagneticAdjointToHzStandard3D(
    TensorView<float> AdjHzPrev,
    TensorView<float> AdjHzMid,
    TensorView<float> HzDecay
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjHzPrev);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjHzPrev.size(1), AdjHzPrev.size(2));
    AdjHzPrev[index.x, index.y, index.z] = AdjHzMid[index.x, index.y, index.z] * HzDecay[index.x, index.y, index.z];
}

[AutoPyBindCUDA]
[CUDAKernel]
void accumulateForwardDiffAdjointX3D(
    TensorView<float> FieldGrad,
    TensorView<float> DiffGrad,
    float invDx
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(FieldGrad);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, FieldGrad.size(1), FieldGrad.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float value = 0.0f;
    if (i < DiffGrad.size(0))
    {
        value -= invDx * DiffGrad[i, j, k];
    }
    if (i > 0)
    {
        uint prevI = i - 1;
        value += invDx * DiffGrad[prevI, j, k];
    }
    FieldGrad[i, j, k] += value;
}

[AutoPyBindCUDA]
[CUDAKernel]
void accumulateForwardDiffAdjointY3D(
    TensorView<float> FieldGrad,
    TensorView<float> DiffGrad,
    float invDy
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(FieldGrad);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, FieldGrad.size(1), FieldGrad.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float value = 0.0f;
    if (j < DiffGrad.size(1))
    {
        value -= invDy * DiffGrad[i, j, k];
    }
    if (j > 0)
    {
        uint prevJ = j - 1;
        value += invDy * DiffGrad[i, prevJ, k];
    }
    FieldGrad[i, j, k] += value;
}

[AutoPyBindCUDA]
[CUDAKernel]
void accumulateForwardDiffAdjointZ3D(
    TensorView<float> FieldGrad,
    TensorView<float> DiffGrad,
    float invDz
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(FieldGrad);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, FieldGrad.size(1), FieldGrad.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float value = 0.0f;
    if (k < DiffGrad.size(2))
    {
        value -= invDz * DiffGrad[i, j, k];
    }
    if (k > 0)
    {
        uint prevK = k - 1;
        value += invDz * DiffGrad[i, j, prevK];
    }
    FieldGrad[i, j, k] += value;
}

[AutoPyBindCUDA]
[CUDAKernel]
void accumulateBackwardDiffAdjointX3D(
    TensorView<float> FieldGrad,
    TensorView<float> DiffGrad,
    float invDx
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(FieldGrad);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, FieldGrad.size(1), FieldGrad.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float value = 0.0f;
    if (i > 0)
    {
        value += invDx * DiffGrad[i, j, k];
    }
    if (i + 1 < FieldGrad.size(0))
    {
        uint nextI = i + 1;
        value -= invDx * DiffGrad[nextI, j, k];
    }
    FieldGrad[i, j, k] += value;
}

[AutoPyBindCUDA]
[CUDAKernel]
void accumulateBackwardDiffAdjointY3D(
    TensorView<float> FieldGrad,
    TensorView<float> DiffGrad,
    float invDy
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(FieldGrad);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, FieldGrad.size(1), FieldGrad.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float value = 0.0f;
    if (j > 0)
    {
        value += invDy * DiffGrad[i, j, k];
    }
    if (j + 1 < FieldGrad.size(1))
    {
        uint nextJ = j + 1;
        value -= invDy * DiffGrad[i, nextJ, k];
    }
    FieldGrad[i, j, k] += value;
}

[AutoPyBindCUDA]
[CUDAKernel]
void accumulateBackwardDiffAdjointZ3D(
    TensorView<float> FieldGrad,
    TensorView<float> DiffGrad,
    float invDz
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(FieldGrad);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, FieldGrad.size(1), FieldGrad.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float value = 0.0f;
    if (k > 0)
    {
        value += invDz * DiffGrad[i, j, k];
    }
    if (k + 1 < FieldGrad.size(2))
    {
        uint nextK = k + 1;
        value -= invDz * DiffGrad[i, j, nextK];
    }
    FieldGrad[i, j, k] += value;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseElectricComponentExCpml3D(
    TensorView<float> AdjExPrev,
    TensorView<float> GradEpsEx,
    TensorView<float> AdjPsiPosPrev,
    TensorView<float> AdjPsiNegPrev,
    TensorView<float> AdjDPos,
    TensorView<float> AdjDNeg,
    TensorView<float> AdjExPost,
    TensorView<float> AdjPsiPosPost,
    TensorView<float> AdjPsiNegPost,
    TensorView<float> ExDecay,
    TensorView<float> ExCurl,
    TensorView<float> EpsEx,
    TensorView<float> PsiPos,
    TensorView<float> PsiNeg,
    TensorView<float> BPos,
    TensorView<float> CPos,
    TensorView<float> InvKappaPos,
    TensorView<float> BNeg,
    TensorView<float> CNeg,
    TensorView<float> InvKappaNeg,
    TensorView<float> HyMid,
    TensorView<float> HzMid,
    float invDy,
    float invDz,
    uint yLowBoundaryMode,
    uint yHighBoundaryMode,
    uint zLowBoundaryMode,
    uint zHighBoundaryMode
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjExPrev);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjExPrev.size(1), AdjExPrev.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    ElectricCellStatus status = resolveElectricCellStatus(
        j,
        AdjExPrev.size(1),
        yLowBoundaryMode,
        yHighBoundaryMode,
        k,
        AdjExPrev.size(2),
        zLowBoundaryMode,
        zHighBoundaryMode
    );

    float adjoint = 0.0f;
    float grad = 0.0f;
    float adjPsiPos = AdjPsiPosPost[i, j, k];
    float adjPsiNeg = AdjPsiNegPost[i, j, k];
    float adjDPos = 0.0f;
    float adjDNeg = 0.0f;

    if (status.inactive)
    {
        adjoint = AdjExPost[i, j, k];
    }
    else if (status.active)
    {
        uint prevJ = j - 1;
        uint prevK = k - 1;
        float dPos = (HzMid[i, j, k] - HzMid[i, prevJ, k]) * invDy;
        float dNeg = (HyMid[i, j, k] - HyMid[i, j, prevK]) * invDz;
        float psiPosCandidate = BPos[j] * PsiPos[i, j, k] + CPos[j] * dPos;
        float psiNegCandidate = BNeg[k] * PsiNeg[i, j, k] + CNeg[k] * dNeg;
        float curlH = (dPos * InvKappaPos[j] + psiPosCandidate) - (dNeg * InvKappaNeg[k] + psiNegCandidate);
        float adjCurlH = AdjExPost[i, j, k] * ExCurl[i, j, k];

        adjoint = AdjExPost[i, j, k] * ExDecay[i, j, k];
        grad = -AdjExPost[i, j, k] * ExCurl[i, j, k] * curlH / EpsEx[i, j, k];
        adjPsiPos = BPos[j] * (AdjPsiPosPost[i, j, k] + adjCurlH);
        adjPsiNeg = BNeg[k] * (AdjPsiNegPost[i, j, k] - adjCurlH);
        adjDPos = InvKappaPos[j] * adjCurlH + CPos[j] * (AdjPsiPosPost[i, j, k] + adjCurlH);
        adjDNeg = -InvKappaNeg[k] * adjCurlH + CNeg[k] * (AdjPsiNegPost[i, j, k] - adjCurlH);
    }

    AdjExPrev[i, j, k] = adjoint;
    GradEpsEx[i, j, k] = grad;
    AdjPsiPosPrev[i, j, k] = adjPsiPos;
    AdjPsiNegPrev[i, j, k] = adjPsiNeg;
    AdjDPos[i, j, k] = adjDPos;
    AdjDNeg[i, j, k] = adjDNeg;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseElectricComponentEyCpml3D(
    TensorView<float> AdjEyPrev,
    TensorView<float> GradEpsEy,
    TensorView<float> AdjPsiPosPrev,
    TensorView<float> AdjPsiNegPrev,
    TensorView<float> AdjDPos,
    TensorView<float> AdjDNeg,
    TensorView<float> AdjEyPost,
    TensorView<float> AdjPsiPosPost,
    TensorView<float> AdjPsiNegPost,
    TensorView<float> EyDecay,
    TensorView<float> EyCurl,
    TensorView<float> EpsEy,
    TensorView<float> PsiPos,
    TensorView<float> PsiNeg,
    TensorView<float> BPos,
    TensorView<float> CPos,
    TensorView<float> InvKappaPos,
    TensorView<float> BNeg,
    TensorView<float> CNeg,
    TensorView<float> InvKappaNeg,
    TensorView<float> HxMid,
    TensorView<float> HzMid,
    float invDx,
    float invDz,
    uint xLowBoundaryMode,
    uint xHighBoundaryMode,
    uint zLowBoundaryMode,
    uint zHighBoundaryMode
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjEyPrev);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjEyPrev.size(1), AdjEyPrev.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    ElectricCellStatus status = resolveElectricCellStatus(
        i,
        AdjEyPrev.size(0),
        xLowBoundaryMode,
        xHighBoundaryMode,
        k,
        AdjEyPrev.size(2),
        zLowBoundaryMode,
        zHighBoundaryMode
    );

    float adjoint = 0.0f;
    float grad = 0.0f;
    float adjPsiPos = AdjPsiPosPost[i, j, k];
    float adjPsiNeg = AdjPsiNegPost[i, j, k];
    float adjDPos = 0.0f;
    float adjDNeg = 0.0f;

    if (status.inactive)
    {
        adjoint = AdjEyPost[i, j, k];
    }
    else if (status.active)
    {
        uint prevI = i - 1;
        uint prevK = k - 1;
        float dPos = (HxMid[i, j, k] - HxMid[i, j, prevK]) * invDz;
        float dNeg = (HzMid[i, j, k] - HzMid[prevI, j, k]) * invDx;
        float psiPosCandidate = BPos[k] * PsiPos[i, j, k] + CPos[k] * dPos;
        float psiNegCandidate = BNeg[i] * PsiNeg[i, j, k] + CNeg[i] * dNeg;
        float curlH = (dPos * InvKappaPos[k] + psiPosCandidate) - (dNeg * InvKappaNeg[i] + psiNegCandidate);
        float adjCurlH = AdjEyPost[i, j, k] * EyCurl[i, j, k];

        adjoint = AdjEyPost[i, j, k] * EyDecay[i, j, k];
        grad = -AdjEyPost[i, j, k] * EyCurl[i, j, k] * curlH / EpsEy[i, j, k];
        adjPsiPos = BPos[k] * (AdjPsiPosPost[i, j, k] + adjCurlH);
        adjPsiNeg = BNeg[i] * (AdjPsiNegPost[i, j, k] - adjCurlH);
        adjDPos = InvKappaPos[k] * adjCurlH + CPos[k] * (AdjPsiPosPost[i, j, k] + adjCurlH);
        adjDNeg = -InvKappaNeg[i] * adjCurlH + CNeg[i] * (AdjPsiNegPost[i, j, k] - adjCurlH);
    }

    AdjEyPrev[i, j, k] = adjoint;
    GradEpsEy[i, j, k] = grad;
    AdjPsiPosPrev[i, j, k] = adjPsiPos;
    AdjPsiNegPrev[i, j, k] = adjPsiNeg;
    AdjDPos[i, j, k] = adjDPos;
    AdjDNeg[i, j, k] = adjDNeg;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseElectricComponentEzCpml3D(
    TensorView<float> AdjEzPrev,
    TensorView<float> GradEpsEz,
    TensorView<float> AdjPsiPosPrev,
    TensorView<float> AdjPsiNegPrev,
    TensorView<float> AdjDPos,
    TensorView<float> AdjDNeg,
    TensorView<float> AdjEzPost,
    TensorView<float> AdjPsiPosPost,
    TensorView<float> AdjPsiNegPost,
    TensorView<float> EzDecay,
    TensorView<float> EzCurl,
    TensorView<float> EpsEz,
    TensorView<float> PsiPos,
    TensorView<float> PsiNeg,
    TensorView<float> BPos,
    TensorView<float> CPos,
    TensorView<float> InvKappaPos,
    TensorView<float> BNeg,
    TensorView<float> CNeg,
    TensorView<float> InvKappaNeg,
    TensorView<float> HxMid,
    TensorView<float> HyMid,
    float invDx,
    float invDy,
    uint xLowBoundaryMode,
    uint xHighBoundaryMode,
    uint yLowBoundaryMode,
    uint yHighBoundaryMode
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjEzPrev);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjEzPrev.size(1), AdjEzPrev.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    ElectricCellStatus status = resolveElectricCellStatus(
        i,
        AdjEzPrev.size(0),
        xLowBoundaryMode,
        xHighBoundaryMode,
        j,
        AdjEzPrev.size(1),
        yLowBoundaryMode,
        yHighBoundaryMode
    );

    float adjoint = 0.0f;
    float grad = 0.0f;
    float adjPsiPos = AdjPsiPosPost[i, j, k];
    float adjPsiNeg = AdjPsiNegPost[i, j, k];
    float adjDPos = 0.0f;
    float adjDNeg = 0.0f;

    if (status.inactive)
    {
        adjoint = AdjEzPost[i, j, k];
    }
    else if (status.active)
    {
        uint prevI = i - 1;
        uint prevJ = j - 1;
        float dPos = (HyMid[i, j, k] - HyMid[prevI, j, k]) * invDx;
        float dNeg = (HxMid[i, j, k] - HxMid[i, prevJ, k]) * invDy;
        float psiPosCandidate = BPos[i] * PsiPos[i, j, k] + CPos[i] * dPos;
        float psiNegCandidate = BNeg[j] * PsiNeg[i, j, k] + CNeg[j] * dNeg;
        float curlH = (dPos * InvKappaPos[i] + psiPosCandidate) - (dNeg * InvKappaNeg[j] + psiNegCandidate);
        float adjCurlH = AdjEzPost[i, j, k] * EzCurl[i, j, k];

        adjoint = AdjEzPost[i, j, k] * EzDecay[i, j, k];
        grad = -AdjEzPost[i, j, k] * EzCurl[i, j, k] * curlH / EpsEz[i, j, k];
        adjPsiPos = BPos[i] * (AdjPsiPosPost[i, j, k] + adjCurlH);
        adjPsiNeg = BNeg[j] * (AdjPsiNegPost[i, j, k] - adjCurlH);
        adjDPos = InvKappaPos[i] * adjCurlH + CPos[i] * (AdjPsiPosPost[i, j, k] + adjCurlH);
        adjDNeg = -InvKappaNeg[j] * adjCurlH + CNeg[j] * (AdjPsiNegPost[i, j, k] - adjCurlH);
    }

    AdjEzPrev[i, j, k] = adjoint;
    GradEpsEz[i, j, k] = grad;
    AdjPsiPosPrev[i, j, k] = adjPsiPos;
    AdjPsiNegPrev[i, j, k] = adjPsiNeg;
    AdjDPos[i, j, k] = adjDPos;
    AdjDNeg[i, j, k] = adjDNeg;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseMagneticComponentHxCpml3D(
    TensorView<float> AdjHxPrev,
    TensorView<float> AdjPsiPosPrev,
    TensorView<float> AdjPsiNegPrev,
    TensorView<float> AdjDPos,
    TensorView<float> AdjDNeg,
    TensorView<float> AdjHxPost,
    TensorView<float> AdjPsiPosPost,
    TensorView<float> AdjPsiNegPost,
    TensorView<float> HxDecay,
    TensorView<float> HxCurl,
    TensorView<float> BPos,
    TensorView<float> CPos,
    TensorView<float> InvKappaPos,
    TensorView<float> BNeg,
    TensorView<float> CNeg,
    TensorView<float> InvKappaNeg
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjHxPrev);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjHxPrev.size(1), AdjHxPrev.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float adjCurlE = -HxCurl[i, j, k] * AdjHxPost[i, j, k];
    float adjPsiPosCandidate = AdjPsiPosPost[i, j, k] + adjCurlE;
    float adjPsiNegCandidate = AdjPsiNegPost[i, j, k] - adjCurlE;

    AdjHxPrev[i, j, k] = AdjHxPost[i, j, k] * HxDecay[i, j, k];
    AdjPsiPosPrev[i, j, k] = BPos[j] * adjPsiPosCandidate;
    AdjPsiNegPrev[i, j, k] = BNeg[k] * adjPsiNegCandidate;
    AdjDPos[i, j, k] = InvKappaPos[j] * adjCurlE + CPos[j] * adjPsiPosCandidate;
    AdjDNeg[i, j, k] = -InvKappaNeg[k] * adjCurlE + CNeg[k] * adjPsiNegCandidate;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseMagneticComponentHyCpml3D(
    TensorView<float> AdjHyPrev,
    TensorView<float> AdjPsiPosPrev,
    TensorView<float> AdjPsiNegPrev,
    TensorView<float> AdjDPos,
    TensorView<float> AdjDNeg,
    TensorView<float> AdjHyPost,
    TensorView<float> AdjPsiPosPost,
    TensorView<float> AdjPsiNegPost,
    TensorView<float> HyDecay,
    TensorView<float> HyCurl,
    TensorView<float> BPos,
    TensorView<float> CPos,
    TensorView<float> InvKappaPos,
    TensorView<float> BNeg,
    TensorView<float> CNeg,
    TensorView<float> InvKappaNeg
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjHyPrev);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjHyPrev.size(1), AdjHyPrev.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float adjCurlE = -HyCurl[i, j, k] * AdjHyPost[i, j, k];
    float adjPsiPosCandidate = AdjPsiPosPost[i, j, k] + adjCurlE;
    float adjPsiNegCandidate = AdjPsiNegPost[i, j, k] - adjCurlE;

    AdjHyPrev[i, j, k] = AdjHyPost[i, j, k] * HyDecay[i, j, k];
    AdjPsiPosPrev[i, j, k] = BPos[k] * adjPsiPosCandidate;
    AdjPsiNegPrev[i, j, k] = BNeg[i] * adjPsiNegCandidate;
    AdjDPos[i, j, k] = InvKappaPos[k] * adjCurlE + CPos[k] * adjPsiPosCandidate;
    AdjDNeg[i, j, k] = -InvKappaNeg[i] * adjCurlE + CNeg[i] * adjPsiNegCandidate;
}

[AutoPyBindCUDA]
[CUDAKernel]
void reverseMagneticComponentHzCpml3D(
    TensorView<float> AdjHzPrev,
    TensorView<float> AdjPsiPosPrev,
    TensorView<float> AdjPsiNegPrev,
    TensorView<float> AdjDPos,
    TensorView<float> AdjDNeg,
    TensorView<float> AdjHzPost,
    TensorView<float> AdjPsiPosPost,
    TensorView<float> AdjPsiNegPost,
    TensorView<float> HzDecay,
    TensorView<float> HzCurl,
    TensorView<float> BPos,
    TensorView<float> CPos,
    TensorView<float> InvKappaPos,
    TensorView<float> BNeg,
    TensorView<float> CNeg,
    TensorView<float> InvKappaNeg
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(AdjHzPrev);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, AdjHzPrev.size(1), AdjHzPrev.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float adjCurlE = -HzCurl[i, j, k] * AdjHzPost[i, j, k];
    float adjPsiPosCandidate = AdjPsiPosPost[i, j, k] + adjCurlE;
    float adjPsiNegCandidate = AdjPsiNegPost[i, j, k] - adjCurlE;

    AdjHzPrev[i, j, k] = AdjHzPost[i, j, k] * HzDecay[i, j, k];
    AdjPsiPosPrev[i, j, k] = BPos[i] * adjPsiPosCandidate;
    AdjPsiNegPrev[i, j, k] = BNeg[j] * adjPsiNegCandidate;
    AdjDPos[i, j, k] = InvKappaPos[i] * adjCurlE + CPos[i] * adjPsiPosCandidate;
    AdjDNeg[i, j, k] = -InvKappaNeg[j] * adjCurlE + CNeg[j] * adjPsiNegCandidate;
}

#endif
