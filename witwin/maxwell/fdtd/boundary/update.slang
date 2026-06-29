#ifndef MAXWELL_FDTD_BOUNDARY_UPDATE_SLANG
#define MAXWELL_FDTD_BOUNDARY_UPDATE_SLANG

#include "helpers.slang"

[AutoPyBindCUDA]
[CUDAKernel]
void updateElectricFieldExCpml3D(
    TensorView<float> Ex,
    TensorView<float> Hy,
    TensorView<float> Hz,
    TensorView<float> ExDecay,
    TensorView<float> ExCurl,
    TensorView<float> PsiExY,
    TensorView<float> PsiExZ,
    TensorView<float> InvKappaExY,
    TensorView<float> BExY,
    TensorView<float> CExY,
    TensorView<float> InvKappaExZ,
    TensorView<float> BExZ,
    TensorView<float> CExZ,
    float invDy,
    float invDz,
    uint yLowBoundaryMode,
    uint yHighBoundaryMode,
    uint zLowBoundaryMode,
    uint zHighBoundaryMode
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Ex);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Ex.size(1), Ex.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    bool yBoundary = isBoundaryIndex(j, Ex.size(1));
    bool zBoundary = isBoundaryIndex(k, Ex.size(2));
    uint yBoundaryMode = yBoundary ? selectBoundaryMode(yLowBoundaryMode, yHighBoundaryMode, j) : BOUNDARY_NONE;
    uint zBoundaryMode = zBoundary ? selectBoundaryMode(zLowBoundaryMode, zHighBoundaryMode, k) : BOUNDARY_NONE;

    if ((yBoundary && isPecBoundaryMode(yBoundaryMode)) || (zBoundary && isPecBoundaryMode(zBoundaryMode)))
    {
        Ex[i, j, k] = 0.0;
        return;
    }

    if ((yBoundary && isInactiveBoundaryMode(yBoundaryMode)) || (zBoundary && isInactiveBoundaryMode(zBoundaryMode)))
    {
        return;
    }

    float dHz_dy = 0.0;
    if (yBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            yBoundaryMode,
            j,
            Hz[i, 0, k],
            Hz[i, Hz.size(1) - 1, k],
            invDy
        );
        if (diff.skip)
        {
            return;
        }
        dHz_dy = diff.value;
    }
    else
    {
        dHz_dy = (Hz[i, j, k] - Hz[i, j - 1, k]) * invDy;
    }

    float dHy_dz = 0.0;
    if (zBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            zBoundaryMode,
            k,
            Hy[i, j, 0],
            Hy[i, j, Hy.size(2) - 1],
            invDz
        );
        if (diff.skip)
        {
            return;
        }
        dHy_dz = diff.value;
    }
    else
    {
        dHy_dz = (Hy[i, j, k] - Hy[i, j, k - 1]) * invDz;
    }

    float psiY = BExY[j] * PsiExY[i, j, k] + CExY[j] * dHz_dy;
    float psiZ = BExZ[k] * PsiExZ[i, j, k] + CExZ[k] * dHy_dz;
    PsiExY[i, j, k] = psiY;
    PsiExZ[i, j, k] = psiZ;

    float curlH = (dHz_dy * InvKappaExY[j] + psiY) - (dHy_dz * InvKappaExZ[k] + psiZ);
    Ex[i, j, k] = Ex[i, j, k] * ExDecay[i, j, k] + ExCurl[i, j, k] * curlH;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateElectricFieldEyCpml3D(
    TensorView<float> Ey,
    TensorView<float> Hx,
    TensorView<float> Hz,
    TensorView<float> EyDecay,
    TensorView<float> EyCurl,
    TensorView<float> PsiEyX,
    TensorView<float> PsiEyZ,
    TensorView<float> InvKappaEyX,
    TensorView<float> BEyX,
    TensorView<float> CEyX,
    TensorView<float> InvKappaEyZ,
    TensorView<float> BEyZ,
    TensorView<float> CEyZ,
    float invDx,
    float invDz,
    uint xLowBoundaryMode,
    uint xHighBoundaryMode,
    uint zLowBoundaryMode,
    uint zHighBoundaryMode
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Ey);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Ey.size(1), Ey.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    bool xBoundary = isBoundaryIndex(i, Ey.size(0));
    bool zBoundary = isBoundaryIndex(k, Ey.size(2));
    uint xBoundaryMode = xBoundary ? selectBoundaryMode(xLowBoundaryMode, xHighBoundaryMode, i) : BOUNDARY_NONE;
    uint zBoundaryMode = zBoundary ? selectBoundaryMode(zLowBoundaryMode, zHighBoundaryMode, k) : BOUNDARY_NONE;

    if ((xBoundary && isPecBoundaryMode(xBoundaryMode)) || (zBoundary && isPecBoundaryMode(zBoundaryMode)))
    {
        Ey[i, j, k] = 0.0;
        return;
    }

    if ((xBoundary && isInactiveBoundaryMode(xBoundaryMode)) || (zBoundary && isInactiveBoundaryMode(zBoundaryMode)))
    {
        return;
    }

    float dHx_dz = 0.0;
    if (zBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            zBoundaryMode,
            k,
            Hx[i, j, 0],
            Hx[i, j, Hx.size(2) - 1],
            invDz
        );
        if (diff.skip)
        {
            return;
        }
        dHx_dz = diff.value;
    }
    else
    {
        dHx_dz = (Hx[i, j, k] - Hx[i, j, k - 1]) * invDz;
    }

    float dHz_dx = 0.0;
    if (xBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            xBoundaryMode,
            i,
            Hz[0, j, k],
            Hz[Hz.size(0) - 1, j, k],
            invDx
        );
        if (diff.skip)
        {
            return;
        }
        dHz_dx = diff.value;
    }
    else
    {
        dHz_dx = (Hz[i, j, k] - Hz[i - 1, j, k]) * invDx;
    }

    float psiX = BEyX[i] * PsiEyX[i, j, k] + CEyX[i] * dHz_dx;
    float psiZ = BEyZ[k] * PsiEyZ[i, j, k] + CEyZ[k] * dHx_dz;
    PsiEyX[i, j, k] = psiX;
    PsiEyZ[i, j, k] = psiZ;

    float curlH = (dHx_dz * InvKappaEyZ[k] + psiZ) - (dHz_dx * InvKappaEyX[i] + psiX);
    Ey[i, j, k] = Ey[i, j, k] * EyDecay[i, j, k] + EyCurl[i, j, k] * curlH;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateElectricFieldEzCpml3D(
    TensorView<float> Ez,
    TensorView<float> Hx,
    TensorView<float> Hy,
    TensorView<float> EzDecay,
    TensorView<float> EzCurl,
    TensorView<float> PsiEzX,
    TensorView<float> PsiEzY,
    TensorView<float> InvKappaEzX,
    TensorView<float> BEzX,
    TensorView<float> CEzX,
    TensorView<float> InvKappaEzY,
    TensorView<float> BEzY,
    TensorView<float> CEzY,
    float invDx,
    float invDy,
    uint xLowBoundaryMode,
    uint xHighBoundaryMode,
    uint yLowBoundaryMode,
    uint yHighBoundaryMode
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Ez);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Ez.size(1), Ez.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    bool xBoundary = isBoundaryIndex(i, Ez.size(0));
    bool yBoundary = isBoundaryIndex(j, Ez.size(1));
    uint xBoundaryMode = xBoundary ? selectBoundaryMode(xLowBoundaryMode, xHighBoundaryMode, i) : BOUNDARY_NONE;
    uint yBoundaryMode = yBoundary ? selectBoundaryMode(yLowBoundaryMode, yHighBoundaryMode, j) : BOUNDARY_NONE;

    if ((xBoundary && isPecBoundaryMode(xBoundaryMode)) || (yBoundary && isPecBoundaryMode(yBoundaryMode)))
    {
        Ez[i, j, k] = 0.0;
        return;
    }

    if ((xBoundary && isInactiveBoundaryMode(xBoundaryMode)) || (yBoundary && isInactiveBoundaryMode(yBoundaryMode)))
    {
        return;
    }

    float dHy_dx = 0.0;
    if (xBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            xBoundaryMode,
            i,
            Hy[0, j, k],
            Hy[Hy.size(0) - 1, j, k],
            invDx
        );
        if (diff.skip)
        {
            return;
        }
        dHy_dx = diff.value;
    }
    else
    {
        dHy_dx = (Hy[i, j, k] - Hy[i - 1, j, k]) * invDx;
    }

    float dHx_dy = 0.0;
    if (yBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            yBoundaryMode,
            j,
            Hx[i, 0, k],
            Hx[i, Hx.size(1) - 1, k],
            invDy
        );
        if (diff.skip)
        {
            return;
        }
        dHx_dy = diff.value;
    }
    else
    {
        dHx_dy = (Hx[i, j, k] - Hx[i, j - 1, k]) * invDy;
    }

    float psiX = BEzX[i] * PsiEzX[i, j, k] + CEzX[i] * dHy_dx;
    float psiY = BEzY[j] * PsiEzY[i, j, k] + CEzY[j] * dHx_dy;
    PsiEzX[i, j, k] = psiX;
    PsiEzY[i, j, k] = psiY;

    float curlH = (dHy_dx * InvKappaEzX[i] + psiX) - (dHx_dy * InvKappaEzY[j] + psiY);
    Ez[i, j, k] = Ez[i, j, k] * EzDecay[i, j, k] + EzCurl[i, j, k] * curlH;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateElectricFieldExCpmlCompressed3D(
    TensorView<float> Ex,
    TensorView<float> Hy,
    TensorView<float> Hz,
    TensorView<float> ExDecay,
    TensorView<float> ExCurl,
    TensorView<float> PsiExY,
    TensorView<float> PsiExZ,
    TensorView<float> InvKappaExY,
    TensorView<float> BExY,
    TensorView<float> CExY,
    TensorView<float> InvKappaExZ,
    TensorView<float> BExZ,
    TensorView<float> CExZ,
    float invDy,
    float invDz,
    uint yLowBoundaryMode,
    uint yHighBoundaryMode,
    uint zLowBoundaryMode,
    uint zHighBoundaryMode,
    uint psiExYLowLength,
    uint psiExYHighStart,
    uint psiExYHighLength,
    uint psiExZLowLength,
    uint psiExZHighStart,
    uint psiExZHighLength
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Ex);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Ex.size(1), Ex.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    bool yBoundary = isBoundaryIndex(j, Ex.size(1));
    bool zBoundary = isBoundaryIndex(k, Ex.size(2));
    uint yBoundaryMode = yBoundary ? selectBoundaryMode(yLowBoundaryMode, yHighBoundaryMode, j) : BOUNDARY_NONE;
    uint zBoundaryMode = zBoundary ? selectBoundaryMode(zLowBoundaryMode, zHighBoundaryMode, k) : BOUNDARY_NONE;

    if ((yBoundary && isPecBoundaryMode(yBoundaryMode)) || (zBoundary && isPecBoundaryMode(zBoundaryMode)))
    {
        Ex[i, j, k] = 0.0;
        return;
    }

    if ((yBoundary && isInactiveBoundaryMode(yBoundaryMode)) || (zBoundary && isInactiveBoundaryMode(zBoundaryMode)))
    {
        return;
    }

    float dHz_dy = 0.0;
    if (yBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            yBoundaryMode,
            j,
            Hz[i, 0, k],
            Hz[i, Hz.size(1) - 1, k],
            invDy
        );
        if (diff.skip)
        {
            return;
        }
        dHz_dy = diff.value;
    }
    else
    {
        dHz_dy = (Hz[i, j, k] - Hz[i, j - 1, k]) * invDy;
    }

    float dHy_dz = 0.0;
    if (zBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            zBoundaryMode,
            k,
            Hy[i, j, 0],
            Hy[i, j, Hy.size(2) - 1],
            invDz
        );
        if (diff.skip)
        {
            return;
        }
        dHy_dz = diff.value;
    }
    else
    {
        dHy_dz = (Hy[i, j, k] - Hy[i, j, k - 1]) * invDz;
    }

    CompressedCpmlIndex psiYIndex = resolveCompressedCpmlIndex(j, psiExYLowLength, psiExYHighStart, psiExYHighLength);
    CompressedCpmlIndex psiZIndex = resolveCompressedCpmlIndex(k, psiExZLowLength, psiExZHighStart, psiExZHighLength);

    float psiY = 0.0;
    if (psiYIndex.active)
    {
        psiY = BExY[j] * PsiExY[i, psiYIndex.index, k] + CExY[j] * dHz_dy;
        PsiExY[i, psiYIndex.index, k] = psiY;
    }
    float psiZ = 0.0;
    if (psiZIndex.active)
    {
        psiZ = BExZ[k] * PsiExZ[i, j, psiZIndex.index] + CExZ[k] * dHy_dz;
        PsiExZ[i, j, psiZIndex.index] = psiZ;
    }

    float curlH = (dHz_dy * InvKappaExY[j] + psiY) - (dHy_dz * InvKappaExZ[k] + psiZ);
    Ex[i, j, k] = Ex[i, j, k] * ExDecay[i, j, k] + ExCurl[i, j, k] * curlH;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateElectricFieldEyCpmlCompressed3D(
    TensorView<float> Ey,
    TensorView<float> Hx,
    TensorView<float> Hz,
    TensorView<float> EyDecay,
    TensorView<float> EyCurl,
    TensorView<float> PsiEyX,
    TensorView<float> PsiEyZ,
    TensorView<float> InvKappaEyX,
    TensorView<float> BEyX,
    TensorView<float> CEyX,
    TensorView<float> InvKappaEyZ,
    TensorView<float> BEyZ,
    TensorView<float> CEyZ,
    float invDx,
    float invDz,
    uint xLowBoundaryMode,
    uint xHighBoundaryMode,
    uint zLowBoundaryMode,
    uint zHighBoundaryMode,
    uint psiEyXLowLength,
    uint psiEyXHighStart,
    uint psiEyXHighLength,
    uint psiEyZLowLength,
    uint psiEyZHighStart,
    uint psiEyZHighLength
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Ey);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Ey.size(1), Ey.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    bool xBoundary = isBoundaryIndex(i, Ey.size(0));
    bool zBoundary = isBoundaryIndex(k, Ey.size(2));
    uint xBoundaryMode = xBoundary ? selectBoundaryMode(xLowBoundaryMode, xHighBoundaryMode, i) : BOUNDARY_NONE;
    uint zBoundaryMode = zBoundary ? selectBoundaryMode(zLowBoundaryMode, zHighBoundaryMode, k) : BOUNDARY_NONE;

    if ((xBoundary && isPecBoundaryMode(xBoundaryMode)) || (zBoundary && isPecBoundaryMode(zBoundaryMode)))
    {
        Ey[i, j, k] = 0.0;
        return;
    }

    if ((xBoundary && isInactiveBoundaryMode(xBoundaryMode)) || (zBoundary && isInactiveBoundaryMode(zBoundaryMode)))
    {
        return;
    }

    float dHx_dz = 0.0;
    if (zBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            zBoundaryMode,
            k,
            Hx[i, j, 0],
            Hx[i, j, Hx.size(2) - 1],
            invDz
        );
        if (diff.skip)
        {
            return;
        }
        dHx_dz = diff.value;
    }
    else
    {
        dHx_dz = (Hx[i, j, k] - Hx[i, j, k - 1]) * invDz;
    }

    float dHz_dx = 0.0;
    if (xBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            xBoundaryMode,
            i,
            Hz[0, j, k],
            Hz[Hz.size(0) - 1, j, k],
            invDx
        );
        if (diff.skip)
        {
            return;
        }
        dHz_dx = diff.value;
    }
    else
    {
        dHz_dx = (Hz[i, j, k] - Hz[i - 1, j, k]) * invDx;
    }

    CompressedCpmlIndex psiXIndex = resolveCompressedCpmlIndex(i, psiEyXLowLength, psiEyXHighStart, psiEyXHighLength);
    CompressedCpmlIndex psiZIndex = resolveCompressedCpmlIndex(k, psiEyZLowLength, psiEyZHighStart, psiEyZHighLength);

    float psiX = 0.0;
    if (psiXIndex.active)
    {
        psiX = BEyX[i] * PsiEyX[psiXIndex.index, j, k] + CEyX[i] * dHz_dx;
        PsiEyX[psiXIndex.index, j, k] = psiX;
    }
    float psiZ = 0.0;
    if (psiZIndex.active)
    {
        psiZ = BEyZ[k] * PsiEyZ[i, j, psiZIndex.index] + CEyZ[k] * dHx_dz;
        PsiEyZ[i, j, psiZIndex.index] = psiZ;
    }

    float curlH = (dHx_dz * InvKappaEyZ[k] + psiZ) - (dHz_dx * InvKappaEyX[i] + psiX);
    Ey[i, j, k] = Ey[i, j, k] * EyDecay[i, j, k] + EyCurl[i, j, k] * curlH;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateElectricFieldEzCpmlCompressed3D(
    TensorView<float> Ez,
    TensorView<float> Hx,
    TensorView<float> Hy,
    TensorView<float> EzDecay,
    TensorView<float> EzCurl,
    TensorView<float> PsiEzX,
    TensorView<float> PsiEzY,
    TensorView<float> InvKappaEzX,
    TensorView<float> BEzX,
    TensorView<float> CEzX,
    TensorView<float> InvKappaEzY,
    TensorView<float> BEzY,
    TensorView<float> CEzY,
    float invDx,
    float invDy,
    uint xLowBoundaryMode,
    uint xHighBoundaryMode,
    uint yLowBoundaryMode,
    uint yHighBoundaryMode,
    uint psiEzXLowLength,
    uint psiEzXHighStart,
    uint psiEzXHighLength,
    uint psiEzYLowLength,
    uint psiEzYHighStart,
    uint psiEzYHighLength
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Ez);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Ez.size(1), Ez.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    bool xBoundary = isBoundaryIndex(i, Ez.size(0));
    bool yBoundary = isBoundaryIndex(j, Ez.size(1));
    uint xBoundaryMode = xBoundary ? selectBoundaryMode(xLowBoundaryMode, xHighBoundaryMode, i) : BOUNDARY_NONE;
    uint yBoundaryMode = yBoundary ? selectBoundaryMode(yLowBoundaryMode, yHighBoundaryMode, j) : BOUNDARY_NONE;

    if ((xBoundary && isPecBoundaryMode(xBoundaryMode)) || (yBoundary && isPecBoundaryMode(yBoundaryMode)))
    {
        Ez[i, j, k] = 0.0;
        return;
    }

    if ((xBoundary && isInactiveBoundaryMode(xBoundaryMode)) || (yBoundary && isInactiveBoundaryMode(yBoundaryMode)))
    {
        return;
    }

    float dHy_dx = 0.0;
    if (xBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            xBoundaryMode,
            i,
            Hy[0, j, k],
            Hy[Hy.size(0) - 1, j, k],
            invDx
        );
        if (diff.skip)
        {
            return;
        }
        dHy_dx = diff.value;
    }
    else
    {
        dHy_dx = (Hy[i, j, k] - Hy[i - 1, j, k]) * invDx;
    }

    float dHx_dy = 0.0;
    if (yBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            yBoundaryMode,
            j,
            Hx[i, 0, k],
            Hx[i, Hx.size(1) - 1, k],
            invDy
        );
        if (diff.skip)
        {
            return;
        }
        dHx_dy = diff.value;
    }
    else
    {
        dHx_dy = (Hx[i, j, k] - Hx[i, j - 1, k]) * invDy;
    }

    CompressedCpmlIndex psiXIndex = resolveCompressedCpmlIndex(i, psiEzXLowLength, psiEzXHighStart, psiEzXHighLength);
    CompressedCpmlIndex psiYIndex = resolveCompressedCpmlIndex(j, psiEzYLowLength, psiEzYHighStart, psiEzYHighLength);

    float psiX = 0.0;
    if (psiXIndex.active)
    {
        psiX = BEzX[i] * PsiEzX[psiXIndex.index, j, k] + CEzX[i] * dHy_dx;
        PsiEzX[psiXIndex.index, j, k] = psiX;
    }
    float psiY = 0.0;
    if (psiYIndex.active)
    {
        psiY = BEzY[j] * PsiEzY[i, psiYIndex.index, k] + CEzY[j] * dHx_dy;
        PsiEzY[i, psiYIndex.index, k] = psiY;
    }

    float curlH = (dHy_dx * InvKappaEzX[i] + psiX) - (dHx_dy * InvKappaEzY[j] + psiY);
    Ez[i, j, k] = Ez[i, j, k] * EzDecay[i, j, k] + EzCurl[i, j, k] * curlH;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateMagneticFieldHxCpmlCompressed3D(
    TensorView<float> Hx,
    TensorView<float> Ey,
    TensorView<float> Ez,
    TensorView<float> HxDecay,
    TensorView<float> HxCurl,
    TensorView<float> PsiHxY,
    TensorView<float> PsiHxZ,
    TensorView<float> InvKappaHxY,
    TensorView<float> ByHxY,
    TensorView<float> CyHxY,
    TensorView<float> InvKappaHxZ,
    TensorView<float> ByHxZ,
    TensorView<float> CyHxZ,
    float invDy,
    float invDz,
    uint psiHxYLowLength,
    uint psiHxYHighStart,
    uint psiHxYHighLength,
    uint psiHxZLowLength,
    uint psiHxZHighStart,
    uint psiHxZHighLength
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Hx);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Hx.size(1), Hx.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float dEz_dy = (Ez[i, j + 1, k] - Ez[i, j, k]) * invDy;
    float dEy_dz = (Ey[i, j, k + 1] - Ey[i, j, k]) * invDz;
    CompressedCpmlIndex psiYIndex = resolveCompressedCpmlIndex(j, psiHxYLowLength, psiHxYHighStart, psiHxYHighLength);
    CompressedCpmlIndex psiZIndex = resolveCompressedCpmlIndex(k, psiHxZLowLength, psiHxZHighStart, psiHxZHighLength);

    float psiY = 0.0;
    if (psiYIndex.active)
    {
        psiY = ByHxY[j] * PsiHxY[i, psiYIndex.index, k] + CyHxY[j] * dEz_dy;
        PsiHxY[i, psiYIndex.index, k] = psiY;
    }
    float psiZ = 0.0;
    if (psiZIndex.active)
    {
        psiZ = ByHxZ[k] * PsiHxZ[i, j, psiZIndex.index] + CyHxZ[k] * dEy_dz;
        PsiHxZ[i, j, psiZIndex.index] = psiZ;
    }

    float curlE = (dEz_dy * InvKappaHxY[j] + psiY) - (dEy_dz * InvKappaHxZ[k] + psiZ);
    Hx[i, j, k] = Hx[i, j, k] * HxDecay[i, j, k] - HxCurl[i, j, k] * curlE;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateMagneticFieldHyCpmlCompressed3D(
    TensorView<float> Hy,
    TensorView<float> Ex,
    TensorView<float> Ez,
    TensorView<float> HyDecay,
    TensorView<float> HyCurl,
    TensorView<float> PsiHyX,
    TensorView<float> PsiHyZ,
    TensorView<float> InvKappaHyX,
    TensorView<float> ByHyX,
    TensorView<float> CyHyX,
    TensorView<float> InvKappaHyZ,
    TensorView<float> ByHyZ,
    TensorView<float> CyHyZ,
    float invDx,
    float invDz,
    uint psiHyXLowLength,
    uint psiHyXHighStart,
    uint psiHyXHighLength,
    uint psiHyZLowLength,
    uint psiHyZHighStart,
    uint psiHyZHighLength
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Hy);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Hy.size(1), Hy.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float dEx_dz = (Ex[i, j, k + 1] - Ex[i, j, k]) * invDz;
    float dEz_dx = (Ez[i + 1, j, k] - Ez[i, j, k]) * invDx;
    CompressedCpmlIndex psiXIndex = resolveCompressedCpmlIndex(i, psiHyXLowLength, psiHyXHighStart, psiHyXHighLength);
    CompressedCpmlIndex psiZIndex = resolveCompressedCpmlIndex(k, psiHyZLowLength, psiHyZHighStart, psiHyZHighLength);

    float psiX = 0.0;
    if (psiXIndex.active)
    {
        psiX = ByHyX[i] * PsiHyX[psiXIndex.index, j, k] + CyHyX[i] * dEz_dx;
        PsiHyX[psiXIndex.index, j, k] = psiX;
    }
    float psiZ = 0.0;
    if (psiZIndex.active)
    {
        psiZ = ByHyZ[k] * PsiHyZ[i, j, psiZIndex.index] + CyHyZ[k] * dEx_dz;
        PsiHyZ[i, j, psiZIndex.index] = psiZ;
    }

    float curlE = (dEx_dz * InvKappaHyZ[k] + psiZ) - (dEz_dx * InvKappaHyX[i] + psiX);
    Hy[i, j, k] = Hy[i, j, k] * HyDecay[i, j, k] - HyCurl[i, j, k] * curlE;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateMagneticFieldHzCpmlCompressed3D(
    TensorView<float> Hz,
    TensorView<float> Ex,
    TensorView<float> Ey,
    TensorView<float> HzDecay,
    TensorView<float> HzCurl,
    TensorView<float> PsiHzX,
    TensorView<float> PsiHzY,
    TensorView<float> InvKappaHzX,
    TensorView<float> ByHzX,
    TensorView<float> CyHzX,
    TensorView<float> InvKappaHzY,
    TensorView<float> ByHzY,
    TensorView<float> CyHzY,
    float invDx,
    float invDy,
    uint psiHzXLowLength,
    uint psiHzXHighStart,
    uint psiHzXHighLength,
    uint psiHzYLowLength,
    uint psiHzYHighStart,
    uint psiHzYHighLength
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Hz);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Hz.size(1), Hz.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    float dEy_dx = (Ey[i + 1, j, k] - Ey[i, j, k]) * invDx;
    float dEx_dy = (Ex[i, j + 1, k] - Ex[i, j, k]) * invDy;
    CompressedCpmlIndex psiXIndex = resolveCompressedCpmlIndex(i, psiHzXLowLength, psiHzXHighStart, psiHzXHighLength);
    CompressedCpmlIndex psiYIndex = resolveCompressedCpmlIndex(j, psiHzYLowLength, psiHzYHighStart, psiHzYHighLength);

    float psiX = 0.0;
    if (psiXIndex.active)
    {
        psiX = ByHzX[i] * PsiHzX[psiXIndex.index, j, k] + CyHzX[i] * dEy_dx;
        PsiHzX[psiXIndex.index, j, k] = psiX;
    }
    float psiY = 0.0;
    if (psiYIndex.active)
    {
        psiY = ByHzY[j] * PsiHzY[i, psiYIndex.index, k] + CyHzY[j] * dEx_dy;
        PsiHzY[i, psiYIndex.index, k] = psiY;
    }

    float curlE = (dEy_dx * InvKappaHzX[i] + psiX) - (dEx_dy * InvKappaHzY[j] + psiY);
    Hz[i, j, k] = Hz[i, j, k] * HzDecay[i, j, k] - HzCurl[i, j, k] * curlE;
}

[AutoPyBindCUDA]
[CUDAKernel]
void applyElectricFieldExCpmlYCorrection3D(
    TensorView<float> Ex,
    TensorView<float> Hz,
    TensorView<float> ExCurl,
    TensorView<float> PsiExY,
    TensorView<float> InvKappaExY,
    TensorView<float> BExY,
    TensorView<float> CExY,
    float invDy,
    uint offsetI,
    uint offsetJ,
    uint offsetK,
    uint zLowBoundaryMode,
    uint zHighBoundaryMode,
    uint fullSizeY,
    uint fullSizeZ
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Ex);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Ex.size(1), Ex.size(2));
    uint gi = index.x + offsetI;
    uint gj = index.y + offsetJ;
    uint gk = index.z + offsetK;
    if (gj == 0 || gj + 1 >= fullSizeY)
    {
        return;
    }
    if (isBoundaryIndex(gk, fullSizeZ))
    {
        uint zBoundaryMode = selectBoundaryMode(zLowBoundaryMode, zHighBoundaryMode, gk);
        if (isPecBoundaryMode(zBoundaryMode) || isInactiveBoundaryMode(zBoundaryMode))
        {
            return;
        }
    }

    float dHz_dy = (Hz[gi, gj, gk] - Hz[gi, gj - 1, gk]) * invDy;
    float psiY = BExY[gj] * PsiExY[index.x, index.y, index.z] + CExY[gj] * dHz_dy;
    PsiExY[index.x, index.y, index.z] = psiY;
    Ex[index.x, index.y, index.z] += ExCurl[index.x, index.y, index.z] * (dHz_dy * (InvKappaExY[gj] - 1.0f) + psiY);
}

[AutoPyBindCUDA]
[CUDAKernel]
void applyElectricFieldExCpmlZCorrection3D(
    TensorView<float> Ex,
    TensorView<float> Hy,
    TensorView<float> ExCurl,
    TensorView<float> PsiExZ,
    TensorView<float> InvKappaExZ,
    TensorView<float> BExZ,
    TensorView<float> CExZ,
    float invDz,
    uint offsetI,
    uint offsetJ,
    uint offsetK,
    uint yLowBoundaryMode,
    uint yHighBoundaryMode,
    uint fullSizeY,
    uint fullSizeZ
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Ex);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Ex.size(1), Ex.size(2));
    uint gi = index.x + offsetI;
    uint gj = index.y + offsetJ;
    uint gk = index.z + offsetK;
    if (gk == 0 || gk + 1 >= fullSizeZ)
    {
        return;
    }
    if (isBoundaryIndex(gj, fullSizeY))
    {
        uint yBoundaryMode = selectBoundaryMode(yLowBoundaryMode, yHighBoundaryMode, gj);
        if (isPecBoundaryMode(yBoundaryMode) || isInactiveBoundaryMode(yBoundaryMode))
        {
            return;
        }
    }

    float dHy_dz = (Hy[gi, gj, gk] - Hy[gi, gj, gk - 1]) * invDz;
    float psiZ = BExZ[gk] * PsiExZ[index.x, index.y, index.z] + CExZ[gk] * dHy_dz;
    PsiExZ[index.x, index.y, index.z] = psiZ;
    Ex[index.x, index.y, index.z] -= ExCurl[index.x, index.y, index.z] * (dHy_dz * (InvKappaExZ[gk] - 1.0f) + psiZ);
}

[AutoPyBindCUDA]
[CUDAKernel]
void applyElectricFieldEyCpmlXCorrection3D(
    TensorView<float> Ey,
    TensorView<float> Hz,
    TensorView<float> EyCurl,
    TensorView<float> PsiEyX,
    TensorView<float> InvKappaEyX,
    TensorView<float> BEyX,
    TensorView<float> CEyX,
    float invDx,
    uint offsetI,
    uint offsetJ,
    uint offsetK,
    uint zLowBoundaryMode,
    uint zHighBoundaryMode,
    uint fullSizeX,
    uint fullSizeZ
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Ey);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Ey.size(1), Ey.size(2));
    uint gi = index.x + offsetI;
    uint gj = index.y + offsetJ;
    uint gk = index.z + offsetK;
    if (gi == 0 || gi + 1 >= fullSizeX)
    {
        return;
    }
    if (isBoundaryIndex(gk, fullSizeZ))
    {
        uint zBoundaryMode = selectBoundaryMode(zLowBoundaryMode, zHighBoundaryMode, gk);
        if (isPecBoundaryMode(zBoundaryMode) || isInactiveBoundaryMode(zBoundaryMode))
        {
            return;
        }
    }

    float dHz_dx = (Hz[gi, gj, gk] - Hz[gi - 1, gj, gk]) * invDx;
    float psiX = BEyX[gi] * PsiEyX[index.x, index.y, index.z] + CEyX[gi] * dHz_dx;
    PsiEyX[index.x, index.y, index.z] = psiX;
    Ey[index.x, index.y, index.z] -= EyCurl[index.x, index.y, index.z] * (dHz_dx * (InvKappaEyX[gi] - 1.0f) + psiX);
}

[AutoPyBindCUDA]
[CUDAKernel]
void applyElectricFieldEyCpmlZCorrection3D(
    TensorView<float> Ey,
    TensorView<float> Hx,
    TensorView<float> EyCurl,
    TensorView<float> PsiEyZ,
    TensorView<float> InvKappaEyZ,
    TensorView<float> BEyZ,
    TensorView<float> CEyZ,
    float invDz,
    uint offsetI,
    uint offsetJ,
    uint offsetK,
    uint xLowBoundaryMode,
    uint xHighBoundaryMode,
    uint fullSizeX,
    uint fullSizeZ
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Ey);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Ey.size(1), Ey.size(2));
    uint gi = index.x + offsetI;
    uint gj = index.y + offsetJ;
    uint gk = index.z + offsetK;
    if (gk == 0 || gk + 1 >= fullSizeZ)
    {
        return;
    }
    if (isBoundaryIndex(gi, fullSizeX))
    {
        uint xBoundaryMode = selectBoundaryMode(xLowBoundaryMode, xHighBoundaryMode, gi);
        if (isPecBoundaryMode(xBoundaryMode) || isInactiveBoundaryMode(xBoundaryMode))
        {
            return;
        }
    }

    float dHx_dz = (Hx[gi, gj, gk] - Hx[gi, gj, gk - 1]) * invDz;
    float psiZ = BEyZ[gk] * PsiEyZ[index.x, index.y, index.z] + CEyZ[gk] * dHx_dz;
    PsiEyZ[index.x, index.y, index.z] = psiZ;
    Ey[index.x, index.y, index.z] += EyCurl[index.x, index.y, index.z] * (dHx_dz * (InvKappaEyZ[gk] - 1.0f) + psiZ);
}

[AutoPyBindCUDA]
[CUDAKernel]
void applyElectricFieldEzCpmlXCorrection3D(
    TensorView<float> Ez,
    TensorView<float> Hy,
    TensorView<float> EzCurl,
    TensorView<float> PsiEzX,
    TensorView<float> InvKappaEzX,
    TensorView<float> BEzX,
    TensorView<float> CEzX,
    float invDx,
    uint offsetI,
    uint offsetJ,
    uint offsetK,
    uint yLowBoundaryMode,
    uint yHighBoundaryMode,
    uint fullSizeX,
    uint fullSizeY
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Ez);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Ez.size(1), Ez.size(2));
    uint gi = index.x + offsetI;
    uint gj = index.y + offsetJ;
    uint gk = index.z + offsetK;
    if (gi == 0 || gi + 1 >= fullSizeX)
    {
        return;
    }
    if (isBoundaryIndex(gj, fullSizeY))
    {
        uint yBoundaryMode = selectBoundaryMode(yLowBoundaryMode, yHighBoundaryMode, gj);
        if (isPecBoundaryMode(yBoundaryMode) || isInactiveBoundaryMode(yBoundaryMode))
        {
            return;
        }
    }

    float dHy_dx = (Hy[gi, gj, gk] - Hy[gi - 1, gj, gk]) * invDx;
    float psiX = BEzX[gi] * PsiEzX[index.x, index.y, index.z] + CEzX[gi] * dHy_dx;
    PsiEzX[index.x, index.y, index.z] = psiX;
    Ez[index.x, index.y, index.z] += EzCurl[index.x, index.y, index.z] * (dHy_dx * (InvKappaEzX[gi] - 1.0f) + psiX);
}

[AutoPyBindCUDA]
[CUDAKernel]
void applyElectricFieldEzCpmlYCorrection3D(
    TensorView<float> Ez,
    TensorView<float> Hx,
    TensorView<float> EzCurl,
    TensorView<float> PsiEzY,
    TensorView<float> InvKappaEzY,
    TensorView<float> BEzY,
    TensorView<float> CEzY,
    float invDy,
    uint offsetI,
    uint offsetJ,
    uint offsetK,
    uint xLowBoundaryMode,
    uint xHighBoundaryMode,
    uint fullSizeX,
    uint fullSizeY
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Ez);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Ez.size(1), Ez.size(2));
    uint gi = index.x + offsetI;
    uint gj = index.y + offsetJ;
    uint gk = index.z + offsetK;
    if (gj == 0 || gj + 1 >= fullSizeY)
    {
        return;
    }
    if (isBoundaryIndex(gi, fullSizeX))
    {
        uint xBoundaryMode = selectBoundaryMode(xLowBoundaryMode, xHighBoundaryMode, gi);
        if (isPecBoundaryMode(xBoundaryMode) || isInactiveBoundaryMode(xBoundaryMode))
        {
            return;
        }
    }

    float dHx_dy = (Hx[gi, gj, gk] - Hx[gi, gj - 1, gk]) * invDy;
    float psiY = BEzY[gj] * PsiEzY[index.x, index.y, index.z] + CEzY[gj] * dHx_dy;
    PsiEzY[index.x, index.y, index.z] = psiY;
    Ez[index.x, index.y, index.z] -= EzCurl[index.x, index.y, index.z] * (dHx_dy * (InvKappaEzY[gj] - 1.0f) + psiY);
}

[AutoPyBindCUDA]
[CUDAKernel]
void applyMagneticFieldHxCpmlYCorrection3D(
    TensorView<float> Hx,
    TensorView<float> Ez,
    TensorView<float> HxCurl,
    TensorView<float> PsiHxY,
    TensorView<float> InvKappaHxY,
    TensorView<float> ByHxY,
    TensorView<float> CyHxY,
    float invDy,
    uint offsetI,
    uint offsetJ,
    uint offsetK
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Hx);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Hx.size(1), Hx.size(2));
    uint gi = index.x + offsetI;
    uint gj = index.y + offsetJ;
    uint gk = index.z + offsetK;
    float dEz_dy = (Ez[gi, gj + 1, gk] - Ez[gi, gj, gk]) * invDy;
    float psiY = ByHxY[gj] * PsiHxY[index.x, index.y, index.z] + CyHxY[gj] * dEz_dy;
    PsiHxY[index.x, index.y, index.z] = psiY;
    Hx[index.x, index.y, index.z] -= HxCurl[index.x, index.y, index.z] * (dEz_dy * (InvKappaHxY[gj] - 1.0f) + psiY);
}

[AutoPyBindCUDA]
[CUDAKernel]
void applyMagneticFieldHxCpmlZCorrection3D(
    TensorView<float> Hx,
    TensorView<float> Ey,
    TensorView<float> HxCurl,
    TensorView<float> PsiHxZ,
    TensorView<float> InvKappaHxZ,
    TensorView<float> ByHxZ,
    TensorView<float> CyHxZ,
    float invDz,
    uint offsetI,
    uint offsetJ,
    uint offsetK
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Hx);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Hx.size(1), Hx.size(2));
    uint gi = index.x + offsetI;
    uint gj = index.y + offsetJ;
    uint gk = index.z + offsetK;
    float dEy_dz = (Ey[gi, gj, gk + 1] - Ey[gi, gj, gk]) * invDz;
    float psiZ = ByHxZ[gk] * PsiHxZ[index.x, index.y, index.z] + CyHxZ[gk] * dEy_dz;
    PsiHxZ[index.x, index.y, index.z] = psiZ;
    Hx[index.x, index.y, index.z] += HxCurl[index.x, index.y, index.z] * (dEy_dz * (InvKappaHxZ[gk] - 1.0f) + psiZ);
}

[AutoPyBindCUDA]
[CUDAKernel]
void applyMagneticFieldHyCpmlXCorrection3D(
    TensorView<float> Hy,
    TensorView<float> Ez,
    TensorView<float> HyCurl,
    TensorView<float> PsiHyX,
    TensorView<float> InvKappaHyX,
    TensorView<float> ByHyX,
    TensorView<float> CyHyX,
    float invDx,
    uint offsetI,
    uint offsetJ,
    uint offsetK
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Hy);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Hy.size(1), Hy.size(2));
    uint gi = index.x + offsetI;
    uint gj = index.y + offsetJ;
    uint gk = index.z + offsetK;
    float dEz_dx = (Ez[gi + 1, gj, gk] - Ez[gi, gj, gk]) * invDx;
    float psiX = ByHyX[gi] * PsiHyX[index.x, index.y, index.z] + CyHyX[gi] * dEz_dx;
    PsiHyX[index.x, index.y, index.z] = psiX;
    Hy[index.x, index.y, index.z] -= HyCurl[index.x, index.y, index.z] * (dEz_dx * (InvKappaHyX[gi] - 1.0f) + psiX);
}

[AutoPyBindCUDA]
[CUDAKernel]
void applyMagneticFieldHyCpmlZCorrection3D(
    TensorView<float> Hy,
    TensorView<float> Ex,
    TensorView<float> HyCurl,
    TensorView<float> PsiHyZ,
    TensorView<float> InvKappaHyZ,
    TensorView<float> ByHyZ,
    TensorView<float> CyHyZ,
    float invDz,
    uint offsetI,
    uint offsetJ,
    uint offsetK
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Hy);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Hy.size(1), Hy.size(2));
    uint gi = index.x + offsetI;
    uint gj = index.y + offsetJ;
    uint gk = index.z + offsetK;
    float dEx_dz = (Ex[gi, gj, gk + 1] - Ex[gi, gj, gk]) * invDz;
    float psiZ = ByHyZ[gk] * PsiHyZ[index.x, index.y, index.z] + CyHyZ[gk] * dEx_dz;
    PsiHyZ[index.x, index.y, index.z] = psiZ;
    Hy[index.x, index.y, index.z] += HyCurl[index.x, index.y, index.z] * (dEx_dz * (InvKappaHyZ[gk] - 1.0f) + psiZ);
}

[AutoPyBindCUDA]
[CUDAKernel]
void applyMagneticFieldHzCpmlXCorrection3D(
    TensorView<float> Hz,
    TensorView<float> Ey,
    TensorView<float> HzCurl,
    TensorView<float> PsiHzX,
    TensorView<float> InvKappaHzX,
    TensorView<float> ByHzX,
    TensorView<float> CyHzX,
    float invDx,
    uint offsetI,
    uint offsetJ,
    uint offsetK
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Hz);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Hz.size(1), Hz.size(2));
    uint gi = index.x + offsetI;
    uint gj = index.y + offsetJ;
    uint gk = index.z + offsetK;
    float dEy_dx = (Ey[gi + 1, gj, gk] - Ey[gi, gj, gk]) * invDx;
    float psiX = ByHzX[gi] * PsiHzX[index.x, index.y, index.z] + CyHzX[gi] * dEy_dx;
    PsiHzX[index.x, index.y, index.z] = psiX;
    Hz[index.x, index.y, index.z] -= HzCurl[index.x, index.y, index.z] * (dEy_dx * (InvKappaHzX[gi] - 1.0f) + psiX);
}

[AutoPyBindCUDA]
[CUDAKernel]
void applyMagneticFieldHzCpmlYCorrection3D(
    TensorView<float> Hz,
    TensorView<float> Ex,
    TensorView<float> HzCurl,
    TensorView<float> PsiHzY,
    TensorView<float> InvKappaHzY,
    TensorView<float> ByHzY,
    TensorView<float> CyHzY,
    float invDy,
    uint offsetI,
    uint offsetJ,
    uint offsetK
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Hz);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Hz.size(1), Hz.size(2));
    uint gi = index.x + offsetI;
    uint gj = index.y + offsetJ;
    uint gk = index.z + offsetK;
    float dEx_dy = (Ex[gi, gj + 1, gk] - Ex[gi, gj, gk]) * invDy;
    float psiY = ByHzY[gj] * PsiHzY[index.x, index.y, index.z] + CyHzY[gj] * dEx_dy;
    PsiHzY[index.x, index.y, index.z] = psiY;
    Hz[index.x, index.y, index.z] += HzCurl[index.x, index.y, index.z] * (dEx_dy * (InvKappaHzY[gj] - 1.0f) + psiY);
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateElectricFieldExStandard3D(
    TensorView<float> Ex,
    TensorView<float> Hy,
    TensorView<float> Hz,
    TensorView<float> ExDecay,
    TensorView<float> ExCurl,
    float invDy,
    float invDz,
    uint yLowBoundaryMode,
    uint yHighBoundaryMode,
    uint zLowBoundaryMode,
    uint zHighBoundaryMode
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Ex);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Ex.size(1), Ex.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    bool yBoundary = isBoundaryIndex(j, Ex.size(1));
    bool zBoundary = isBoundaryIndex(k, Ex.size(2));
    uint yBoundaryMode = yBoundary ? selectBoundaryMode(yLowBoundaryMode, yHighBoundaryMode, j) : BOUNDARY_NONE;
    uint zBoundaryMode = zBoundary ? selectBoundaryMode(zLowBoundaryMode, zHighBoundaryMode, k) : BOUNDARY_NONE;

    if ((yBoundary && isPecBoundaryMode(yBoundaryMode)) || (zBoundary && isPecBoundaryMode(zBoundaryMode)))
    {
        Ex[i, j, k] = 0.0;
        return;
    }

    if ((yBoundary && isInactiveBoundaryMode(yBoundaryMode)) || (zBoundary && isInactiveBoundaryMode(zBoundaryMode)))
    {
        return;
    }

    float dHz_dy = 0.0;
    if (yBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            yBoundaryMode,
            j,
            Hz[i, 0, k],
            Hz[i, Hz.size(1) - 1, k],
            invDy
        );
        if (diff.skip)
        {
            return;
        }
        dHz_dy = diff.value;
    }
    else
    {
        dHz_dy = (Hz[i, j, k] - Hz[i, j - 1, k]) * invDy;
    }

    float dHy_dz = 0.0;
    if (zBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            zBoundaryMode,
            k,
            Hy[i, j, 0],
            Hy[i, j, Hy.size(2) - 1],
            invDz
        );
        if (diff.skip)
        {
            return;
        }
        dHy_dz = diff.value;
    }
    else
    {
        dHy_dz = (Hy[i, j, k] - Hy[i, j, k - 1]) * invDz;
    }

    float curlH = dHz_dy - dHy_dz;
    Ex[i, j, k] = Ex[i, j, k] * ExDecay[i, j, k] + ExCurl[i, j, k] * curlH;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateElectricFieldEyStandard3D(
    TensorView<float> Ey,
    TensorView<float> Hx,
    TensorView<float> Hz,
    TensorView<float> EyDecay,
    TensorView<float> EyCurl,
    float invDx,
    float invDz,
    uint xLowBoundaryMode,
    uint xHighBoundaryMode,
    uint zLowBoundaryMode,
    uint zHighBoundaryMode
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Ey);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Ey.size(1), Ey.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    bool xBoundary = isBoundaryIndex(i, Ey.size(0));
    bool zBoundary = isBoundaryIndex(k, Ey.size(2));
    uint xBoundaryMode = xBoundary ? selectBoundaryMode(xLowBoundaryMode, xHighBoundaryMode, i) : BOUNDARY_NONE;
    uint zBoundaryMode = zBoundary ? selectBoundaryMode(zLowBoundaryMode, zHighBoundaryMode, k) : BOUNDARY_NONE;

    if ((xBoundary && isPecBoundaryMode(xBoundaryMode)) || (zBoundary && isPecBoundaryMode(zBoundaryMode)))
    {
        Ey[i, j, k] = 0.0;
        return;
    }

    if ((xBoundary && isInactiveBoundaryMode(xBoundaryMode)) || (zBoundary && isInactiveBoundaryMode(zBoundaryMode)))
    {
        return;
    }

    float dHx_dz = 0.0;
    if (zBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            zBoundaryMode,
            k,
            Hx[i, j, 0],
            Hx[i, j, Hx.size(2) - 1],
            invDz
        );
        if (diff.skip)
        {
            return;
        }
        dHx_dz = diff.value;
    }
    else
    {
        dHx_dz = (Hx[i, j, k] - Hx[i, j, k - 1]) * invDz;
    }

    float dHz_dx = 0.0;
    if (xBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            xBoundaryMode,
            i,
            Hz[0, j, k],
            Hz[Hz.size(0) - 1, j, k],
            invDx
        );
        if (diff.skip)
        {
            return;
        }
        dHz_dx = diff.value;
    }
    else
    {
        dHz_dx = (Hz[i, j, k] - Hz[i - 1, j, k]) * invDx;
    }

    float curlH = dHx_dz - dHz_dx;
    Ey[i, j, k] = Ey[i, j, k] * EyDecay[i, j, k] + EyCurl[i, j, k] * curlH;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateElectricFieldEzStandard3D(
    TensorView<float> Ez,
    TensorView<float> Hx,
    TensorView<float> Hy,
    TensorView<float> EzDecay,
    TensorView<float> EzCurl,
    float invDx,
    float invDy,
    uint xLowBoundaryMode,
    uint xHighBoundaryMode,
    uint yLowBoundaryMode,
    uint yHighBoundaryMode
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(Ez);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, Ez.size(1), Ez.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    bool xBoundary = isBoundaryIndex(i, Ez.size(0));
    bool yBoundary = isBoundaryIndex(j, Ez.size(1));
    uint xBoundaryMode = xBoundary ? selectBoundaryMode(xLowBoundaryMode, xHighBoundaryMode, i) : BOUNDARY_NONE;
    uint yBoundaryMode = yBoundary ? selectBoundaryMode(yLowBoundaryMode, yHighBoundaryMode, j) : BOUNDARY_NONE;

    if ((xBoundary && isPecBoundaryMode(xBoundaryMode)) || (yBoundary && isPecBoundaryMode(yBoundaryMode)))
    {
        Ez[i, j, k] = 0.0;
        return;
    }

    if ((xBoundary && isInactiveBoundaryMode(xBoundaryMode)) || (yBoundary && isInactiveBoundaryMode(yBoundaryMode)))
    {
        return;
    }

    float dHy_dx = 0.0;
    if (xBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            xBoundaryMode,
            i,
            Hy[0, j, k],
            Hy[Hy.size(0) - 1, j, k],
            invDx
        );
        if (diff.skip)
        {
            return;
        }
        dHy_dx = diff.value;
    }
    else
    {
        dHy_dx = (Hy[i, j, k] - Hy[i - 1, j, k]) * invDx;
    }

    float dHx_dy = 0.0;
    if (yBoundary)
    {
        DiffFloat diff = boundaryBackwardDiff(
            yBoundaryMode,
            j,
            Hx[i, 0, k],
            Hx[i, Hx.size(1) - 1, k],
            invDy
        );
        if (diff.skip)
        {
            return;
        }
        dHx_dy = diff.value;
    }
    else
    {
        dHx_dy = (Hx[i, j, k] - Hx[i, j - 1, k]) * invDy;
    }

    float curlH = dHy_dx - dHx_dy;
    Ez[i, j, k] = Ez[i, j, k] * EzDecay[i, j, k] + EzCurl[i, j, k] * curlH;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateElectricFieldExBloch3D(
    TensorView<float> ExReal,
    TensorView<float> ExImag,
    TensorView<float> HyReal,
    TensorView<float> HyImag,
    TensorView<float> HzReal,
    TensorView<float> HzImag,
    TensorView<float> ExDecay,
    TensorView<float> ExCurl,
    float phaseCosY,
    float phaseSinY,
    float phaseCosZ,
    float phaseSinZ,
    float invDy,
    float invDz
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(ExReal);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, ExReal.size(1), ExReal.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    bool yBoundary = isBoundaryIndex(j, ExReal.size(1));
    bool zBoundary = isBoundaryIndex(k, ExReal.size(2));

    float2 dHz_dy = float2(0.0, 0.0);
    if (yBoundary)
    {
        DiffFloat2 diff = blochBackwardDiff(
            j,
            float2(HzReal[i, 0, k], HzImag[i, 0, k]),
            float2(HzReal[i, HzReal.size(1) - 1, k], HzImag[i, HzImag.size(1) - 1, k]),
            phaseCosY,
            phaseSinY,
            invDy
        );
        dHz_dy = diff.value;
    }
    else
    {
        dHz_dy = float2(
            (HzReal[i, j, k] - HzReal[i, j - 1, k]) * invDy,
            (HzImag[i, j, k] - HzImag[i, j - 1, k]) * invDy
        );
    }

    float2 dHy_dz = float2(0.0, 0.0);
    if (zBoundary)
    {
        DiffFloat2 diff = blochBackwardDiff(
            k,
            float2(HyReal[i, j, 0], HyImag[i, j, 0]),
            float2(HyReal[i, j, HyReal.size(2) - 1], HyImag[i, j, HyImag.size(2) - 1]),
            phaseCosZ,
            phaseSinZ,
            invDz
        );
        dHy_dz = diff.value;
    }
    else
    {
        dHy_dz = float2(
            (HyReal[i, j, k] - HyReal[i, j, k - 1]) * invDz,
            (HyImag[i, j, k] - HyImag[i, j, k - 1]) * invDz
        );
    }

    float2 curlH = dHz_dy - dHy_dz;
    ExReal[i, j, k] = ExReal[i, j, k] * ExDecay[i, j, k] + ExCurl[i, j, k] * curlH.x;
    ExImag[i, j, k] = ExImag[i, j, k] * ExDecay[i, j, k] + ExCurl[i, j, k] * curlH.y;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateElectricFieldExBlochYStandardZ3D(
    TensorView<float> ExReal,
    TensorView<float> ExImag,
    TensorView<float> HyReal,
    TensorView<float> HyImag,
    TensorView<float> HzReal,
    TensorView<float> HzImag,
    TensorView<float> ExDecay,
    TensorView<float> ExCurl,
    float phaseCosY,
    float phaseSinY,
    float invDy,
    float invDz,
    uint zLowBoundaryMode,
    uint zHighBoundaryMode
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(ExReal);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, ExReal.size(1), ExReal.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    bool yBoundary = isBoundaryIndex(j, ExReal.size(1));
    bool zBoundary = isBoundaryIndex(k, ExReal.size(2));
    uint zBoundaryMode = zBoundary ? selectBoundaryMode(zLowBoundaryMode, zHighBoundaryMode, k) : BOUNDARY_NONE;
    if (zBoundary && isPecBoundaryMode(zBoundaryMode))
    {
        ExReal[i, j, k] = 0.0;
        ExImag[i, j, k] = 0.0;
        return;
    }
    if (zBoundary && isInactiveBoundaryMode(zBoundaryMode))
    {
        return;
    }

    float2 dHz_dy = float2(0.0, 0.0);
    if (yBoundary)
    {
        DiffFloat2 diff = blochBackwardDiff(
            j,
            float2(HzReal[i, 0, k], HzImag[i, 0, k]),
            float2(HzReal[i, HzReal.size(1) - 1, k], HzImag[i, HzImag.size(1) - 1, k]),
            phaseCosY,
            phaseSinY,
            invDy
        );
        dHz_dy = diff.value;
    }
    else
    {
        dHz_dy = float2(
            (HzReal[i, j, k] - HzReal[i, j - 1, k]) * invDy,
            (HzImag[i, j, k] - HzImag[i, j - 1, k]) * invDy
        );
    }

    float2 dHy_dz = float2(0.0, 0.0);
    if (zBoundary)
    {
        DiffFloat diffReal = boundaryBackwardDiff(
            zBoundaryMode,
            k,
            HyReal[i, j, 0],
            HyReal[i, j, HyReal.size(2) - 1],
            invDz
        );
        if (diffReal.skip)
        {
            return;
        }
        DiffFloat diffImag = boundaryBackwardDiff(
            zBoundaryMode,
            k,
            HyImag[i, j, 0],
            HyImag[i, j, HyImag.size(2) - 1],
            invDz
        );
        dHy_dz = float2(diffReal.value, diffImag.value);
    }
    else
    {
        dHy_dz = float2(
            (HyReal[i, j, k] - HyReal[i, j, k - 1]) * invDz,
            (HyImag[i, j, k] - HyImag[i, j, k - 1]) * invDz
        );
    }

    float2 curlH = dHz_dy - dHy_dz;
    ExReal[i, j, k] = ExReal[i, j, k] * ExDecay[i, j, k] + ExCurl[i, j, k] * curlH.x;
    ExImag[i, j, k] = ExImag[i, j, k] * ExDecay[i, j, k] + ExCurl[i, j, k] * curlH.y;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateElectricFieldEyBloch3D(
    TensorView<float> EyReal,
    TensorView<float> EyImag,
    TensorView<float> HxReal,
    TensorView<float> HxImag,
    TensorView<float> HzReal,
    TensorView<float> HzImag,
    TensorView<float> EyDecay,
    TensorView<float> EyCurl,
    float phaseCosX,
    float phaseSinX,
    float phaseCosZ,
    float phaseSinZ,
    float invDx,
    float invDz
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(EyReal);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, EyReal.size(1), EyReal.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    bool xBoundary = isBoundaryIndex(i, EyReal.size(0));
    bool zBoundary = isBoundaryIndex(k, EyReal.size(2));

    float2 dHx_dz = float2(0.0, 0.0);
    if (zBoundary)
    {
        DiffFloat2 diff = blochBackwardDiff(
            k,
            float2(HxReal[i, j, 0], HxImag[i, j, 0]),
            float2(HxReal[i, j, HxReal.size(2) - 1], HxImag[i, j, HxImag.size(2) - 1]),
            phaseCosZ,
            phaseSinZ,
            invDz
        );
        dHx_dz = diff.value;
    }
    else
    {
        dHx_dz = float2(
            (HxReal[i, j, k] - HxReal[i, j, k - 1]) * invDz,
            (HxImag[i, j, k] - HxImag[i, j, k - 1]) * invDz
        );
    }

    float2 dHz_dx = float2(0.0, 0.0);
    if (xBoundary)
    {
        DiffFloat2 diff = blochBackwardDiff(
            i,
            float2(HzReal[0, j, k], HzImag[0, j, k]),
            float2(HzReal[HzReal.size(0) - 1, j, k], HzImag[HzImag.size(0) - 1, j, k]),
            phaseCosX,
            phaseSinX,
            invDx
        );
        dHz_dx = diff.value;
    }
    else
    {
        dHz_dx = float2(
            (HzReal[i, j, k] - HzReal[i - 1, j, k]) * invDx,
            (HzImag[i, j, k] - HzImag[i - 1, j, k]) * invDx
        );
    }

    float2 curlH = dHx_dz - dHz_dx;
    EyReal[i, j, k] = EyReal[i, j, k] * EyDecay[i, j, k] + EyCurl[i, j, k] * curlH.x;
    EyImag[i, j, k] = EyImag[i, j, k] * EyDecay[i, j, k] + EyCurl[i, j, k] * curlH.y;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateElectricFieldEyBlochXStandardZ3D(
    TensorView<float> EyReal,
    TensorView<float> EyImag,
    TensorView<float> HxReal,
    TensorView<float> HxImag,
    TensorView<float> HzReal,
    TensorView<float> HzImag,
    TensorView<float> EyDecay,
    TensorView<float> EyCurl,
    float phaseCosX,
    float phaseSinX,
    float invDx,
    float invDz,
    uint zLowBoundaryMode,
    uint zHighBoundaryMode
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(EyReal);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, EyReal.size(1), EyReal.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    bool xBoundary = isBoundaryIndex(i, EyReal.size(0));
    bool zBoundary = isBoundaryIndex(k, EyReal.size(2));
    uint zBoundaryMode = zBoundary ? selectBoundaryMode(zLowBoundaryMode, zHighBoundaryMode, k) : BOUNDARY_NONE;
    if (zBoundary && isPecBoundaryMode(zBoundaryMode))
    {
        EyReal[i, j, k] = 0.0;
        EyImag[i, j, k] = 0.0;
        return;
    }
    if (zBoundary && isInactiveBoundaryMode(zBoundaryMode))
    {
        return;
    }

    float2 dHx_dz = float2(0.0, 0.0);
    if (zBoundary)
    {
        DiffFloat diffReal = boundaryBackwardDiff(
            zBoundaryMode,
            k,
            HxReal[i, j, 0],
            HxReal[i, j, HxReal.size(2) - 1],
            invDz
        );
        if (diffReal.skip)
        {
            return;
        }
        DiffFloat diffImag = boundaryBackwardDiff(
            zBoundaryMode,
            k,
            HxImag[i, j, 0],
            HxImag[i, j, HxImag.size(2) - 1],
            invDz
        );
        dHx_dz = float2(diffReal.value, diffImag.value);
    }
    else
    {
        dHx_dz = float2(
            (HxReal[i, j, k] - HxReal[i, j, k - 1]) * invDz,
            (HxImag[i, j, k] - HxImag[i, j, k - 1]) * invDz
        );
    }

    float2 dHz_dx = float2(0.0, 0.0);
    if (xBoundary)
    {
        DiffFloat2 diff = blochBackwardDiff(
            i,
            float2(HzReal[0, j, k], HzImag[0, j, k]),
            float2(HzReal[HzReal.size(0) - 1, j, k], HzImag[HzImag.size(0) - 1, j, k]),
            phaseCosX,
            phaseSinX,
            invDx
        );
        dHz_dx = diff.value;
    }
    else
    {
        dHz_dx = float2(
            (HzReal[i, j, k] - HzReal[i - 1, j, k]) * invDx,
            (HzImag[i, j, k] - HzImag[i - 1, j, k]) * invDx
        );
    }

    float2 curlH = dHx_dz - dHz_dx;
    EyReal[i, j, k] = EyReal[i, j, k] * EyDecay[i, j, k] + EyCurl[i, j, k] * curlH.x;
    EyImag[i, j, k] = EyImag[i, j, k] * EyDecay[i, j, k] + EyCurl[i, j, k] * curlH.y;
}

[AutoPyBindCUDA]
[CUDAKernel]
void updateElectricFieldEzBloch3D(
    TensorView<float> EzReal,
    TensorView<float> EzImag,
    TensorView<float> HxReal,
    TensorView<float> HxImag,
    TensorView<float> HyReal,
    TensorView<float> HyImag,
    TensorView<float> EzDecay,
    TensorView<float> EzCurl,
    float phaseCosX,
    float phaseSinX,
    float phaseCosY,
    float phaseSinY,
    float invDx,
    float invDy
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(EzReal);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, EzReal.size(1), EzReal.size(2));
    uint i = index.x;
    uint j = index.y;
    uint k = index.z;

    bool xBoundary = isBoundaryIndex(i, EzReal.size(0));
    bool yBoundary = isBoundaryIndex(j, EzReal.size(1));

    float2 dHy_dx = float2(0.0, 0.0);
    if (xBoundary)
    {
        DiffFloat2 diff = blochBackwardDiff(
            i,
            float2(HyReal[0, j, k], HyImag[0, j, k]),
            float2(HyReal[HyReal.size(0) - 1, j, k], HyImag[HyImag.size(0) - 1, j, k]),
            phaseCosX,
            phaseSinX,
            invDx
        );
        dHy_dx = diff.value;
    }
    else
    {
        dHy_dx = float2(
            (HyReal[i, j, k] - HyReal[i - 1, j, k]) * invDx,
            (HyImag[i, j, k] - HyImag[i - 1, j, k]) * invDx
        );
    }

    float2 dHx_dy = float2(0.0, 0.0);
    if (yBoundary)
    {
        DiffFloat2 diff = blochBackwardDiff(
            j,
            float2(HxReal[i, 0, k], HxImag[i, 0, k]),
            float2(HxReal[i, HxReal.size(1) - 1, k], HxImag[i, HxImag.size(1) - 1, k]),
            phaseCosY,
            phaseSinY,
            invDy
        );
        dHx_dy = diff.value;
    }
    else
    {
        dHx_dy = float2(
            (HxReal[i, j, k] - HxReal[i, j - 1, k]) * invDy,
            (HxImag[i, j, k] - HxImag[i, j - 1, k]) * invDy
        );
    }

    float2 curlH = dHy_dx - dHx_dy;
    EzReal[i, j, k] = EzReal[i, j, k] * EzDecay[i, j, k] + EzCurl[i, j, k] * curlH.x;
    EzImag[i, j, k] = EzImag[i, j, k] * EzDecay[i, j, k] + EzCurl[i, j, k] * curlH.y;
}

#endif
