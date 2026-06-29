#ifndef MAXWELL_FDTD_KERNELS_SOURCES_SLANG
#define MAXWELL_FDTD_KERNELS_SOURCES_SLANG

[AutoPyBindCUDA]
[CUDAKernel]
void addSourcePatchExPeriodic3D(
    TensorView<float> Ex,
    TensorView<float> sourcePatch,
    int offsetI,
    int offsetJ,
    int offsetK,
    float signal,
    uint wrapAxisA,
    uint wrapAxisB
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(sourcePatch);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, sourcePatch.size(1), sourcePatch.size(2));
    int i = offsetI + int(index.x);
    int j = offsetJ + int(index.y);
    int k = offsetK + int(index.z);
    if (i < 0 || i >= int(Ex.size(0)) || j < 0 || j >= int(Ex.size(1)) || k < 0 || k >= int(Ex.size(2)))
    {
        return;
    }

    float delta = signal * sourcePatch[index.x, index.y, index.z];
    Ex[uint(i), uint(j), uint(k)] += delta;

    bool yBoundary = wrapAxisA != 0 && (j == 0 || j + 1 >= int(Ex.size(1)));
    bool zBoundary = wrapAxisB != 0 && (k == 0 || k + 1 >= int(Ex.size(2)));
    int pairJ = (j == 0) ? int(Ex.size(1)) - 1 : 0;
    int pairK = (k == 0) ? int(Ex.size(2)) - 1 : 0;

    if (yBoundary)
    {
        Ex[uint(i), uint(pairJ), uint(k)] += delta;
    }
    if (zBoundary)
    {
        Ex[uint(i), uint(j), uint(pairK)] += delta;
    }
    if (yBoundary && zBoundary)
    {
        Ex[uint(i), uint(pairJ), uint(pairK)] += delta;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void addSourcePatchEyPeriodic3D(
    TensorView<float> Ey,
    TensorView<float> sourcePatch,
    int offsetI,
    int offsetJ,
    int offsetK,
    float signal,
    uint wrapAxisA,
    uint wrapAxisB
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(sourcePatch);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, sourcePatch.size(1), sourcePatch.size(2));
    int i = offsetI + int(index.x);
    int j = offsetJ + int(index.y);
    int k = offsetK + int(index.z);
    if (i < 0 || i >= int(Ey.size(0)) || j < 0 || j >= int(Ey.size(1)) || k < 0 || k >= int(Ey.size(2)))
    {
        return;
    }

    float delta = signal * sourcePatch[index.x, index.y, index.z];
    Ey[uint(i), uint(j), uint(k)] += delta;

    bool xBoundary = wrapAxisA != 0 && (i == 0 || i + 1 >= int(Ey.size(0)));
    bool zBoundary = wrapAxisB != 0 && (k == 0 || k + 1 >= int(Ey.size(2)));
    int pairI = (i == 0) ? int(Ey.size(0)) - 1 : 0;
    int pairK = (k == 0) ? int(Ey.size(2)) - 1 : 0;

    if (xBoundary)
    {
        Ey[uint(pairI), uint(j), uint(k)] += delta;
    }
    if (zBoundary)
    {
        Ey[uint(i), uint(j), uint(pairK)] += delta;
    }
    if (xBoundary && zBoundary)
    {
        Ey[uint(pairI), uint(j), uint(pairK)] += delta;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void addSourcePatchEzPeriodic3D(
    TensorView<float> Ez,
    TensorView<float> sourcePatch,
    int offsetI,
    int offsetJ,
    int offsetK,
    float signal,
    uint wrapAxisA,
    uint wrapAxisB
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(sourcePatch);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, sourcePatch.size(1), sourcePatch.size(2));
    int i = offsetI + int(index.x);
    int j = offsetJ + int(index.y);
    int k = offsetK + int(index.z);
    if (i < 0 || i >= int(Ez.size(0)) || j < 0 || j >= int(Ez.size(1)) || k < 0 || k >= int(Ez.size(2)))
    {
        return;
    }

    float delta = signal * sourcePatch[index.x, index.y, index.z];
    Ez[uint(i), uint(j), uint(k)] += delta;

    bool xBoundary = wrapAxisA != 0 && (i == 0 || i + 1 >= int(Ez.size(0)));
    bool yBoundary = wrapAxisB != 0 && (j == 0 || j + 1 >= int(Ez.size(1)));
    int pairI = (i == 0) ? int(Ez.size(0)) - 1 : 0;
    int pairJ = (j == 0) ? int(Ez.size(1)) - 1 : 0;

    if (xBoundary)
    {
        Ez[uint(pairI), uint(j), uint(k)] += delta;
    }
    if (yBoundary)
    {
        Ez[uint(i), uint(pairJ), uint(k)] += delta;
    }
    if (xBoundary && yBoundary)
    {
        Ez[uint(pairI), uint(pairJ), uint(k)] += delta;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void addSourcePatchBloch3D(
    TensorView<float> ExReal,
    TensorView<float> ExImag,
    TensorView<float> EyReal,
    TensorView<float> EyImag,
    TensorView<float> EzReal,
    TensorView<float> EzImag,
    TensorView<float> sourcePatch,
    int offsetI,
    int offsetJ,
    int offsetK,
    float signalReal,
    float signalImag,
    uint axisCode,
    float phaseCosA,
    float phaseSinA,
    float phaseCosB,
    float phaseSinB,
    uint wrapAxisA,
    uint wrapAxisB
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(sourcePatch);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, sourcePatch.size(1), sourcePatch.size(2));
    int i = offsetI + int(index.x);
    int j = offsetJ + int(index.y);
    int k = offsetK + int(index.z);
    float2 delta = float2(signalReal, signalImag) * sourcePatch[index.x, index.y, index.z];

    if (axisCode == 0)
    {
        if (i < 0 || i >= int(ExReal.size(0)) || j < 0 || j >= int(ExReal.size(1)) || k < 0 || k >= int(ExReal.size(2)))
        {
            return;
        }

        ExReal[uint(i), uint(j), uint(k)] += delta.x;
        ExImag[uint(i), uint(j), uint(k)] += delta.y;

        bool boundaryA = wrapAxisA != 0 && (j == 0 || j + 1 >= int(ExReal.size(1)));
        bool boundaryB = wrapAxisB != 0 && (k == 0 || k + 1 >= int(ExReal.size(2)));
        int pairA = (j == 0) ? int(ExReal.size(1)) - 1 : 0;
        int pairB = (k == 0) ? int(ExReal.size(2)) - 1 : 0;

        if (boundaryA)
        {
            float2 deltaA = (j == 0)
                ? complexPhasePositive(phaseCosA, phaseSinA, delta)
                : complexPhaseNegative(phaseCosA, phaseSinA, delta);
            ExReal[uint(i), uint(pairA), uint(k)] += deltaA.x;
            ExImag[uint(i), uint(pairA), uint(k)] += deltaA.y;
        }
        if (boundaryB)
        {
            float2 deltaB = (k == 0)
                ? complexPhasePositive(phaseCosB, phaseSinB, delta)
                : complexPhaseNegative(phaseCosB, phaseSinB, delta);
            ExReal[uint(i), uint(j), uint(pairB)] += deltaB.x;
            ExImag[uint(i), uint(j), uint(pairB)] += deltaB.y;
        }
        if (boundaryA && boundaryB)
        {
            float2 deltaAB = (j == 0)
                ? complexPhasePositive(phaseCosA, phaseSinA, delta)
                : complexPhaseNegative(phaseCosA, phaseSinA, delta);
            deltaAB = (k == 0)
                ? complexPhasePositive(phaseCosB, phaseSinB, deltaAB)
                : complexPhaseNegative(phaseCosB, phaseSinB, deltaAB);
            ExReal[uint(i), uint(pairA), uint(pairB)] += deltaAB.x;
            ExImag[uint(i), uint(pairA), uint(pairB)] += deltaAB.y;
        }
        return;
    }

    if (axisCode == 1)
    {
        if (i < 0 || i >= int(EyReal.size(0)) || j < 0 || j >= int(EyReal.size(1)) || k < 0 || k >= int(EyReal.size(2)))
        {
            return;
        }

        EyReal[uint(i), uint(j), uint(k)] += delta.x;
        EyImag[uint(i), uint(j), uint(k)] += delta.y;

        bool boundaryA = wrapAxisA != 0 && (i == 0 || i + 1 >= int(EyReal.size(0)));
        bool boundaryB = wrapAxisB != 0 && (k == 0 || k + 1 >= int(EyReal.size(2)));
        int pairA = (i == 0) ? int(EyReal.size(0)) - 1 : 0;
        int pairB = (k == 0) ? int(EyReal.size(2)) - 1 : 0;

        if (boundaryA)
        {
            float2 deltaA = (i == 0)
                ? complexPhasePositive(phaseCosA, phaseSinA, delta)
                : complexPhaseNegative(phaseCosA, phaseSinA, delta);
            EyReal[uint(pairA), uint(j), uint(k)] += deltaA.x;
            EyImag[uint(pairA), uint(j), uint(k)] += deltaA.y;
        }
        if (boundaryB)
        {
            float2 deltaB = (k == 0)
                ? complexPhasePositive(phaseCosB, phaseSinB, delta)
                : complexPhaseNegative(phaseCosB, phaseSinB, delta);
            EyReal[uint(i), uint(j), uint(pairB)] += deltaB.x;
            EyImag[uint(i), uint(j), uint(pairB)] += deltaB.y;
        }
        if (boundaryA && boundaryB)
        {
            float2 deltaAB = (i == 0)
                ? complexPhasePositive(phaseCosA, phaseSinA, delta)
                : complexPhaseNegative(phaseCosA, phaseSinA, delta);
            deltaAB = (k == 0)
                ? complexPhasePositive(phaseCosB, phaseSinB, deltaAB)
                : complexPhaseNegative(phaseCosB, phaseSinB, deltaAB);
            EyReal[uint(pairA), uint(j), uint(pairB)] += deltaAB.x;
            EyImag[uint(pairA), uint(j), uint(pairB)] += deltaAB.y;
        }
        return;
    }

    if (i < 0 || i >= int(EzReal.size(0)) || j < 0 || j >= int(EzReal.size(1)) || k < 0 || k >= int(EzReal.size(2)))
    {
        return;
    }

    EzReal[uint(i), uint(j), uint(k)] += delta.x;
    EzImag[uint(i), uint(j), uint(k)] += delta.y;

    bool boundaryA = wrapAxisA != 0 && (i == 0 || i + 1 >= int(EzReal.size(0)));
    bool boundaryB = wrapAxisB != 0 && (j == 0 || j + 1 >= int(EzReal.size(1)));
    int pairA = (i == 0) ? int(EzReal.size(0)) - 1 : 0;
    int pairB = (j == 0) ? int(EzReal.size(1)) - 1 : 0;

    if (boundaryA)
    {
        float2 deltaA = (i == 0)
            ? complexPhasePositive(phaseCosA, phaseSinA, delta)
            : complexPhaseNegative(phaseCosA, phaseSinA, delta);
        EzReal[uint(pairA), uint(j), uint(k)] += deltaA.x;
        EzImag[uint(pairA), uint(j), uint(k)] += deltaA.y;
    }
    if (boundaryB)
    {
        float2 deltaB = (j == 0)
            ? complexPhasePositive(phaseCosB, phaseSinB, delta)
            : complexPhaseNegative(phaseCosB, phaseSinB, delta);
        EzReal[uint(i), uint(pairB), uint(k)] += deltaB.x;
        EzImag[uint(i), uint(pairB), uint(k)] += deltaB.y;
    }
    if (boundaryA && boundaryB)
    {
        float2 deltaAB = (i == 0)
            ? complexPhasePositive(phaseCosA, phaseSinA, delta)
            : complexPhaseNegative(phaseCosA, phaseSinA, delta);
        deltaAB = (j == 0)
            ? complexPhasePositive(phaseCosB, phaseSinB, deltaAB)
            : complexPhaseNegative(phaseCosB, phaseSinB, deltaAB);
        EzReal[uint(pairA), uint(pairB), uint(k)] += deltaAB.x;
        EzImag[uint(pairA), uint(pairB), uint(k)] += deltaAB.y;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void addSourcePatch3D(
    TensorView<float> field,
    TensorView<float> sourcePatch,
    uint offsetI,
    uint offsetJ,
    uint offsetK,
    float signal
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(sourcePatch);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, sourcePatch.size(1), sourcePatch.size(2));
    field[index.x + offsetI, index.y + offsetJ, index.z + offsetK] += signal * sourcePatch[index.x, index.y, index.z];
}

[AutoPyBindCUDA]
[CUDAKernel]
void addScaledSliceSourcePatch3D(
    TensorView<float> field,
    TensorView<float> sourcePatch,
    TensorView<float> incidentField,
    uint sampleIndex,
    uint offsetI,
    uint offsetJ,
    uint offsetK,
    float scale
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(sourcePatch);
    if (linearIndex >= total)
    {
        return;
    }

    float incident = incidentField[sampleIndex];
    uint3 index = unflatten3D(linearIndex, sourcePatch.size(1), sourcePatch.size(2));
    field[index.x + offsetI, index.y + offsetJ, index.z + offsetK] +=
        scale * incident * sourcePatch[index.x, index.y, index.z];
}

[AutoPyBindCUDA]
[CUDAKernel]
void addScaledLineSourcePatch3D(
    TensorView<float> field,
    TensorView<float> coeffPatch,
    TensorView<float> incidentField,
    TensorView<int> sampleIndices,
    uint sampleAxisCode,
    uint offsetI,
    uint offsetJ,
    uint offsetK,
    float scale
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(coeffPatch);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, coeffPatch.size(1), coeffPatch.size(2));
    uint sampleLinearIndex = sampleAxisCode == 0 ? index.x : (sampleAxisCode == 1 ? index.y : index.z);
    uint sampleIndex = uint(sampleIndices[sampleLinearIndex]);
    float incident = incidentField[sampleIndex];
    field[index.x + offsetI, index.y + offsetJ, index.z + offsetK] +=
        scale * incident * coeffPatch[index.x, index.y, index.z];
}

[AutoPyBindCUDA]
[CUDAKernel]
void addInterpolatedSourcePatch3D(
    TensorView<float> field,
    TensorView<float> coeffPatch,
    TensorView<float> incidentField,
    TensorView<float> samplePositions,
    float origin,
    float ds,
    uint offsetI,
    uint offsetJ,
    uint offsetK,
    float scale
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(coeffPatch);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, coeffPatch.size(1), coeffPatch.size(2));
    uint lastIndex = incidentField.size(0) > 0 ? incidentField.size(0) - 1 : 0;
    float coord = ds > 0.0f ? (samplePositions[index.x, index.y, index.z] - origin) / ds : 0.0f;
    coord = max(coord, 0.0f);
    coord = min(coord, float(lastIndex));
    uint lower = uint(floor(coord));
    uint upper = min(lower + 1, lastIndex);
    float frac = coord - float(lower);
    float incident = incidentField[lower] + frac * (incidentField[upper] - incidentField[lower]);
    field[index.x + offsetI, index.y + offsetJ, index.z + offsetK] +=
        scale * incident * coeffPatch[index.x, index.y, index.z];
}

uint resolveBatchedTermIndex(
    uint linearIndex,
    TensorView<int> termStarts,
    uint totalCount
)
{
    uint termCount = termStarts.size(0);
    for (uint termIndex = 0; termIndex < termCount; ++termIndex)
    {
        uint termStart = uint(termStarts[termIndex]);
        uint termEnd = termIndex + 1 < termCount ? uint(termStarts[termIndex + 1]) : totalCount;
        if (linearIndex < termEnd)
        {
            return termIndex;
        }
    }
    return termCount;
}

[AutoPyBindCUDA]
[CUDAKernel]
void addBatchedReferenceSourcePatches3D(
    TensorView<float> fieldX,
    TensorView<float> fieldY,
    TensorView<float> fieldZ,
    TensorView<float> coeffData,
    TensorView<float> incidentField,
    TensorView<int> termStarts,
    TensorView<int> termShapes,
    TensorView<int> termOffsets,
    TensorView<int> fieldCodes,
    TensorView<int> sampleAxisCodes,
    TensorView<int> sampleIndexStarts,
    TensorView<int> sampleIndices
)
{
    uint linearIndex = linearThreadIndex1D();
    uint totalCount = coeffData.size(0);
    if (linearIndex >= totalCount)
    {
        return;
    }

    uint termIndex = resolveBatchedTermIndex(linearIndex, termStarts, totalCount);
    if (termIndex >= termStarts.size(0))
    {
        return;
    }

    uint termStart = uint(termStarts[termIndex]);
    uint localLinearIndex = linearIndex - termStart;
    uint sizeY = uint(termShapes[termIndex, 1]);
    uint sizeZ = uint(termShapes[termIndex, 2]);
    uint3 localIndex = unflatten3D(localLinearIndex, sizeY, sizeZ);
    uint sampleAxisCode = uint(sampleAxisCodes[termIndex]);
    uint sampleLinearIndex = sampleAxisCode == 0 ? localIndex.x : (sampleAxisCode == 1 ? localIndex.y : localIndex.z);
    uint sampleIndex = uint(sampleIndices[uint(sampleIndexStarts[termIndex]) + sampleLinearIndex]);
    float delta = coeffData[linearIndex] * incidentField[sampleIndex];

    uint fieldI = localIndex.x + uint(termOffsets[termIndex, 0]);
    uint fieldJ = localIndex.y + uint(termOffsets[termIndex, 1]);
    uint fieldK = localIndex.z + uint(termOffsets[termIndex, 2]);
    uint fieldCode = uint(fieldCodes[termIndex]);
    if (fieldCode == 0)
    {
        fieldX[fieldI, fieldJ, fieldK] += delta;
    }
    else if (fieldCode == 1)
    {
        fieldY[fieldI, fieldJ, fieldK] += delta;
    }
    else
    {
        fieldZ[fieldI, fieldJ, fieldK] += delta;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void addBatchedInterpolatedSourcePatches3D(
    TensorView<float> fieldX,
    TensorView<float> fieldY,
    TensorView<float> fieldZ,
    TensorView<float> coeffData,
    TensorView<float> incidentField,
    TensorView<float> samplePositions,
    TensorView<int> termStarts,
    TensorView<int> termShapes,
    TensorView<int> termOffsets,
    TensorView<int> fieldCodes,
    float origin,
    float ds
)
{
    uint linearIndex = linearThreadIndex1D();
    uint totalCount = coeffData.size(0);
    if (linearIndex >= totalCount)
    {
        return;
    }

    uint termIndex = resolveBatchedTermIndex(linearIndex, termStarts, totalCount);
    if (termIndex >= termStarts.size(0))
    {
        return;
    }

    uint termStart = uint(termStarts[termIndex]);
    uint localLinearIndex = linearIndex - termStart;
    uint sizeY = uint(termShapes[termIndex, 1]);
    uint sizeZ = uint(termShapes[termIndex, 2]);
    uint3 localIndex = unflatten3D(localLinearIndex, sizeY, sizeZ);

    uint lastIndex = incidentField.size(0) > 0 ? incidentField.size(0) - 1 : 0;
    float coord = ds > 0.0f ? (samplePositions[linearIndex] - origin) / ds : 0.0f;
    coord = max(coord, 0.0f);
    coord = min(coord, float(lastIndex));
    uint lower = uint(floor(coord));
    uint upper = min(lower + 1, lastIndex);
    float frac = coord - float(lower);
    float incident = incidentField[lower] + frac * (incidentField[upper] - incidentField[lower]);
    float delta = coeffData[linearIndex] * incident;

    uint fieldI = localIndex.x + uint(termOffsets[termIndex, 0]);
    uint fieldJ = localIndex.y + uint(termOffsets[termIndex, 1]);
    uint fieldK = localIndex.z + uint(termOffsets[termIndex, 2]);
    uint fieldCode = uint(fieldCodes[termIndex]);
    if (fieldCode == 0)
    {
        fieldX[fieldI, fieldJ, fieldK] += delta;
    }
    else if (fieldCode == 1)
    {
        fieldY[fieldI, fieldJ, fieldK] += delta;
    }
    else
    {
        fieldZ[fieldI, fieldJ, fieldK] += delta;
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void addCwPhasedSourcePatch3D(
    TensorView<float> field,
    TensorView<float> sourcePatchCos,
    TensorView<float> sourcePatchSin,
    uint offsetI,
    uint offsetJ,
    uint offsetK,
    float signalCos,
    float signalSin
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(sourcePatchCos);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, sourcePatchCos.size(1), sourcePatchCos.size(2));
    field[index.x + offsetI, index.y + offsetJ, index.z + offsetK] +=
        signalCos * sourcePatchCos[index.x, index.y, index.z]
        + signalSin * sourcePatchSin[index.x, index.y, index.z];
}

[AutoPyBindCUDA]
[CUDAKernel]
void addTimeShiftedSourcePatch3D(
    TensorView<float> field,
    TensorView<float> sourcePatch,
    TensorView<float> delayPatch,
    TensorView<float> activationDelayPatch,
    uint offsetI,
    uint offsetJ,
    uint offsetK,
    uint timeKind,
    float time,
    float frequency,
    float fwidth,
    float amplitude,
    float phase,
    float delay,
    uint causalGate
)
{
    uint linearIndex = linearThreadIndex1D();
    uint total = linearSize3D(sourcePatch);
    if (linearIndex >= total)
    {
        return;
    }

    uint3 index = unflatten3D(linearIndex, sourcePatch.size(1), sourcePatch.size(2));
    if (causalGate != 0 && time < activationDelayPatch[index.x, index.y, index.z])
    {
        return;
    }
    float sampleTime = time - delayPatch[index.x, index.y, index.z];
    float signal = evaluateSourceTimeSample(timeKind, sampleTime, frequency, fwidth, amplitude, phase, delay);
    field[index.x + offsetI, index.y + offsetJ, index.z + offsetK] += signal * sourcePatch[index.x, index.y, index.z];
}

#endif
