"""Move references to the 0.6 top-level names into their 0.7.0 namespaces.

0.7.0 phase 5b. The top level of ``witwin.maxwell`` keeps the ``Scene -> Simulation -> Result``
core; every other public object has one home in a namespace (``mw.rf``, ``mw.media.fit``, ...).
The script rewrites references mechanically (``ast`` positions, layout kept):

* ``mw.PortData`` -> ``mw.rf.PortData`` for every module alias of ``witwin.maxwell``
  (``import witwin.maxwell as mw``) and for ``witwin.maxwell.PortData`` after a plain
  ``import witwin.maxwell``;
* ``from witwin.maxwell import PortData, Scene`` -> ``from witwin.maxwell import Scene`` plus
  ``from witwin.maxwell.rf import PortData`` (one statement per namespace);
* ``getattr/hasattr/setattr/delattr(mw, "PortData", ...)`` and ``"PortData" in mw.__all__``
  -> the same on ``mw.rf``; the string ``"witwin.maxwell.PortData"`` -> ``"witwin.maxwell.rf.PortData"``;
* ``from witwin.maxwell.fdtd import FDTD, calculate_required_steps`` and
  ``from witwin.maxwell.fdfd import FDFD`` -> the internal ``.solver`` modules (the namespaces
  no longer export the solver backends).

The map is read from the namespaces themselves: a 0.6 top-level name (``OLD_TOP_LEVEL``) that
``witwin.maxwell.__all__`` no longer lists moves to the one namespace whose ``__all__`` lists it.
Removed names (``mw.run``: use ``Simulation.run()``; the ``BJT``/``MOSFET`` placeholders) are
reported, as are references the script cannot rewrite safely: other uses of ``mw.__all__``,
``getattr(mw, <expression>)``, imports of a name whose namespace is not an importable module
path, and ``mw.<moved>`` spelled inside other strings (code in strings, messages).

Markdown files: fenced ``python`` blocks that parse are rewritten like Python files, and
``mw.<moved>`` in the prose outside them is rewritten textually.

Commands (run with the maxwell environment; the script imports ``witwin.maxwell`` from this
checkout)::

    classify [PATH ...]   per-kind counts of rewrite and review sites, then the review list
    apply [PATH ...]      rewrite in place (idempotent; re-run after a rebase)
    check [PATH ...]      exit 1 while a rewritable reference or a removed name remains
    table                 the old -> new table (CHANGELOG / release notes)

Without paths the scope is every tracked ``.py`` and ``.md`` file under ``tests``,
``benchmark``, ``scripts``, ``examples``, ``experiments``, ``docs/reference``, ``docs/memory``,
``docs/README.md``, ``README.md``, ``FEATURE_LIST.md`` and ``witwin``. Excluded: the MoM line's
``witwin/maxwell/mom/`` and ``tests/mom/`` (pass ``tests/mom`` explicitly to include it),
historical records (``docs/assessments``, ``docs/archive``, ``docs/plans``, ``docs/releases``,
``docs/reference/release-notes-*``, ``CHANGELOG.md``), the codemods, the BITID probes that
also drive frozen 0.6 trees through ``tests/support/api_dialect.py`` (and the dialect itself),
and the consumer-metadata bridge with its pinned loaders. Paths outside this checkout (a user's
own code) are walked directly.
"""
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import types
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = "witwin.maxwell"

# ``witwin.maxwell.__all__`` before phase 5b (after p4c).
OLD_TOP_LEVEL = frozenset("""
    AVERAGING_PROFILE AbsorberKind AdaptiveSweep AdaptiveSweepResult AggregatedSceneGradient
    AngleProjectionMonitor AntennaData ArrayBasisData AstigmaticGaussianBeam
    AstigmaticGaussianOverlapMonitor AuxFieldTimeMonitor AxisPath AxisSegment BJT BasebandGaussianPulse
    BasebandRectangularPulse BatchExecution BatchItem BatchOutput BatchParameters BatchPlanReport
    BatchResult BatchResultIterator BatchSimulation BatchVariation BeamCodebook BeamData BeamWeights
    BendCoordinates BoundaryKind BoundarySpec Box BreakdownEvent BreakdownMonitor BreakdownResultData
    BreakdownStressData BroadbandPulse CPMLConfig CPMLMemoryMode CW CalderonPreconditioner
    CapacitanceData Capacitor CapacityError CartesianProjectionMonitor ChargeDensity CircleLoop Circuit
    CircuitData CircuitDevice CircuitNode ClosedSurfaceMonitor CoaxialLumpedPort CoaxialLumpedResistor
    ComplexPolySlab ComplexTensor3x3 ComponentRating ComponentStressData ComponentStressMonitor
    CompositeCurrentIntegral CompositeVoltageIntegral Cone ConvergenceError CopperFoil CornerFinderSpec
    CurrentControlledCurrentSource CurrentControlledVoltageSource CurrentDataset CurrentSource
    CustomCurrentIntegral CustomCurrentSource CustomDebyePole CustomDrudePole CustomFieldSource
    CustomLorentzPole CustomSourceTime CustomVoltageIntegral Cylinder DebyePole DiagonalTensor3
    DielectricBreakdown Difference DiffractionMonitor Diode DipoleEmissionMonitor DirectH2Config
    DirectionalFMMConfig DiscreteStateSpaceNetwork DispersionFit DispersionFitReport DistributedFailure
    Domain DrudePole DyadicIntegrationConfig ESDCurrentSource ESDDiagnostics ESDPortRecord
    ESDResampledWaveform ESDVoltageSource ESDWaveform EdgeGrading ElectrostaticBoundarySpec
    ElectrostaticInitialCondition ElectrostaticResultData ElectrostaticSolverConfig
    ElectrostaticTerminal Ellipsoid EmbeddedElementPatternData EmbeddedNetworkData ExecutionRecord
    ExperimentalWarning FDFDConfig FDFDSolverConfig FDTDConfig FDTDCoreFallbackWarning FDTDMaterialCache
    FDTDParallelConfig FDTDPassivityError FDTDResumeCheckpoint FMMConfig FailureKind FarFieldData
    FarFieldProjection FieldData FieldDataset FieldFrame FieldProjection FieldQueryConfig
    FieldTimeMonitor FinitePlaneMonitor FitReport FluxData FluxMonitor FluxTimeMonitor FramedFieldData
    FunctionRightPreconditioner GaussianBeam GaussianOverlapMonitor GaussianPulse Graphene
    GridRefinement GridSpec GroisseRoughness GyromagneticFerrite H2Config HMatrixConfig HalfSpace
    HammerstadRoughness HarmonicBalanceConfig HarmonicCutoffConfig HollowBox HurayRoughness
    IdentityRightPreconditioner IncidentPowerDensity IncidentPowerDensityMonitor Inductor
    InternalAbsorber Intersection KSpaceProjectionMonitor KernelErrorBudget Layer LayerMap
    LayerRefinementSpec LayerStack LayeredSurfaceConfig LeafBlockRightPreconditioner LinearCoupledConfig
    LocalDesignRegion LocalResponseConfig LorentzPole LossyMetalMedium LowFrequencySmoothingSpec Ludwig3
    LumpedCurrentSource LumpedPort MATERIAL_LIBRARY MIMOData MNAConfig MOSFET Material
    MaterialLibraryEntry MaterialRegion MaxHoldComposite MaxwellError MaxwellPlaneModes MaxwellWarning
    MeasuredWaveform Medium2D MediumMonitor Mesh MeshOverrideStructure MoMConfig MoMDyadicPeriodicConfig
    MoMPeriodicConfig ModalPortSweep ModeABC ModeAmplitudeData ModeMonitor ModePort ModeSource
    ModulationSpec MultiGPUExecution MultipathEnvironment MurABC MutualInductor
    NearFieldRightPreconditioner NetworkBlock NetworkData NetworkFitReport NetworkPhysicalityReport
    NonlinearSolveConfig NonlinearSusceptibility OpenSurfaceConfig PECFrame PFFTConfig ParallelRLC
    PassivityReport PermittivityMonitor PerturbationMedium PiecewiseLinearIV PiecewiseLinearWaveform
    PlanarPEC PlaneMonitor PlaneWave PointDipole PointMonitor PolySlab PolygonLoop PolylinePath
    PolynomialIV PortBinding PortData PortDrive PortExcitation PortSweep PowerLossData PowerLossMonitor
    PowerNormalization PreconditionerContext PreparedBatch Prism ProcessStack PulseWaveform Pyramid
    RationalFitConfig RationalModel RationalSurfaceImpedance RectangleLoop Resistor Result
    ResultSequence RickerWavelet SARAveraging SARPeak SARResult SParameterData Scene SceneModule
    SchemaError SeriesRLC SetupError Simulation SimulationMethod SineWaveform SmoothBreakdownRisk
    SmoothBreakdownRiskData SpectralSampler SpectralWindowKind Sphere StateSpaceNetwork Structure
    SubpixelSpec SurfaceFieldTimeMonitor SurfaceImpedanceMedium SurfaceImpedanceModel SurfaceMesh
    SurfaceMeshConfig SurfaceRoughness TFSF Tensor3x3 TensorNonlinearSusceptibility TerminalPort
    TerminalRef TerminalWavePort TerminalWavePortApertureWarning TetrahedralVolumeMesh ThinWire
    TimeConfig TimedSwitch Torus TouchstoneNetwork TouchstoneParseError TruncationEstimate
    TruncationReport TwoPhotonAbsorption UniformCurrentSource Union UnsupportedFeatureError ViaMap
    VoltageControlledCurrentSource VoltageControlledVoltageSource VoltageDependentCapacitor
    VoltageSource WaveModeSpec WavePort WireConductor WireData WireEnd WireMonitor WireNodeRef
    WirePortBinding aggregate_scene_gradient_vjp bilinear_discretize capabilities
    check_sampled_passivity combine_coherent_sar combine_incoherent_sar ensemble_scene_gradient_vjp
    fit_dispersion fit_dispersion_nk fit_rational invdes library_material load_dispersion_csv
    parse_spice permittivity_from_nk port_phasors port_voltage postprocess run run_many
    run_modal_port_sweep smooth_low_frequency trapezoidal_impedance
""".split())

# Top-level names that branches landing next to p5b add (api070/p4d-es-config); they move like the
# 0.6 names once a namespace lists them, and are skipped while none does.
PARALLEL_TOP_LEVEL = frozenset({"ElectrostaticConfig", "CapacitanceConfig"})

REMOVED = {
    "run": "removed; call Simulation.run()",
    "BJT": "removed (a reserved placeholder that always raised)",
    "MOSFET": "removed (a reserved placeholder that always raised)",
    # p5c replaced the constant with an enum member.
    "AVERAGING_PROFILE": "removed; use mw.sar.AveragingProfile.CUBICAL_PREFIX_V1",
}
# Solver backends the fdtd/fdfd namespaces no longer export: (module, name) -> internal module.
BACKENDS = {
    (f"{PACKAGE}.fdtd", "FDTD"): f"{PACKAGE}.fdtd.solver",
    (f"{PACKAGE}.fdtd", "calculate_required_steps"): f"{PACKAGE}.fdtd.solver",
    (f"{PACKAGE}.fdfd", "FDFD"): f"{PACKAGE}.fdfd.solver",
}
ATTRIBUTE_CALLS = frozenset({"getattr", "hasattr", "setattr", "delattr"})

SCOPE = ("tests", "benchmark", "scripts", "examples", "experiments", "docs/reference", "docs/memory",
         "docs/README.md", "README.md", "FEATURE_LIST.md", "witwin")
EXCLUDE = re.compile(
    r"^(witwin/maxwell/mom/|tests/mom/|docs/(assessments|archive|plans|releases)/|docs/reference/release-notes-"
    r"|CHANGELOG\.md$|scripts/dev/codemods/|tests/support/api_dialect\.py$"
    r"|tests/fdtd/_bit_identity_probe\.py$|tests/fdtd/test_raw_phase_bit_identity\.py$"
    r"|tests/boundaries/absorbers/_boundary_state_probe\.py$|tests/monitors/in_simulation/_bit_identity_runner\.py$"
    r"|tests/materials/surface_impedance/smooth_surface_probe\.py$|tests/support/rf_port_bit_identity_scene\.py$"
    r"|tests/support/rf_port_bit_identity_terminal_wave_scene\.py$|tests/support/port_step_graph_bit_identity_scene\.py$"
    r"|scripts/dev/fdtd/consumer_metadata/|scripts/dev/measure_(mom|sie)_fmm_scaling\.py$"
    r"|benchmark/sie_single_buffer_equivalence_v3\.py$)")
EXPLICIT = re.compile(r"^tests/mom/")
FENCE = re.compile(r"^(?P<indent>[ \t]*)```(?P<lang>[A-Za-z0-9_+-]*)[^\n]*\n(?P<body>.*?)^(?P=indent)```",
                   re.MULTILINE | re.DOTALL)


@dataclass(frozen=True)
class Home:
    namespace: str    # dotted below the top level: "rf", "media.fit"
    module: str       # module that holds the namespace (importable when it equals the public path)

    @property
    def importable(self) -> bool:
        return self.module == f"{PACKAGE}.{self.namespace}"


@dataclass
class Site:
    path: str
    line: int
    kind: str          # attribute | import | string | dynamic | removed | removed-name | markdown | parse
    status: str        # rewrite | review
    detail: str = ""


# --- the map ----------------------------------------------------------------------------
def load_homes() -> dict[str, Home]:
    """Moved 0.6 top-level name -> its namespace, read from the namespaces' ``__all__``."""
    sys.path.insert(0, str(ROOT))
    import witwin.maxwell as mw

    found: dict[str, list[Home]] = defaultdict(list)

    def walk(module: types.ModuleType, dotted: str) -> None:
        for name in module.__all__:
            value = getattr(module, name)
            if isinstance(value, types.ModuleType):
                walk(value, f"{dotted}.{name}")
            else:
                found[name].append(Home(dotted, module.__name__))

    for namespace in mw._NAMESPACES:
        module = getattr(mw, namespace)
        if hasattr(module, "__all__"):
            walk(module, namespace)
    homes = {}
    for name in sorted((OLD_TOP_LEVEL | PARALLEL_TOP_LEVEL) - set(mw.__all__) - set(REMOVED)):
        if name in mw._NAMESPACES:
            continue
        places = found.get(name, [])
        if not places and name in PARALLEL_TOP_LEVEL:
            continue
        if len(places) != 1:
            raise RuntimeError(f"{name}: expected one namespace home, found {places}")
        homes[name] = places[0]
    return homes


# --- source positions -------------------------------------------------------------------
class Source:
    """Text with ``ast`` (line, UTF-8 byte column) -> character offset conversion."""

    def __init__(self, text: str):
        self.text = text
        self.starts = [0]
        for line in text.splitlines(keepends=True):
            self.starts.append(self.starts[-1] + len(line))
        self.lines = text.splitlines(keepends=True)

    def offset(self, line: int, col: int) -> int:
        raw = self.lines[line - 1].encode("utf-8")[:col]
        return self.starts[line - 1] + len(raw.decode("utf-8"))

    def span(self, node) -> tuple[int, int]:
        return self.offset(node.lineno, node.col_offset), self.offset(node.end_lineno, node.end_col_offset)


def apply_edits(text: str, edits: list[tuple[int, int, str]]) -> str:
    edits = sorted(set(edits))
    for (s0, e0, _), (s1, _e1, _) in zip(edits, edits[1:]):
        if s1 < e0:
            raise RuntimeError(f"overlapping edits at {s0}..{e0} and {s1}")
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    return text


# --- python -----------------------------------------------------------------------------
def module_aliases(tree) -> tuple[set[str], bool]:
    """(names bound to ``witwin.maxwell``, whether ``import witwin.maxwell`` binds ``witwin``)."""
    aliases, plain = set(), False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == PACKAGE:
                    if alias.asname:
                        aliases.add(alias.asname)
                    else:
                        plain = True
        elif isinstance(node, ast.ImportFrom) and node.module == "witwin" and node.level == 0:
            for alias in node.names:
                if alias.name == "maxwell":
                    aliases.add(alias.asname or "maxwell")
    return aliases, plain


def is_package(node, aliases: set[str], plain: bool) -> bool:
    if isinstance(node, ast.Name):
        return node.id in aliases
    return (plain and isinstance(node, ast.Attribute) and node.attr == "maxwell"
            and isinstance(node.value, ast.Name) and node.value.id == "witwin")


def import_statement(module: str, aliases: list[ast.alias]) -> str:
    names = ", ".join(alias.name + (f" as {alias.asname}" if alias.asname else "") for alias in aliases)
    return f"from {module} import {names}"


def rewrite_python(text: str, path: str, homes: dict[str, Home]) -> tuple[str, list[Site]]:
    tree = ast.parse(text)
    source = Source(text)
    aliases, plain = module_aliases(tree)
    edits, sites = [], []

    def site(node, kind, status, detail=""):
        sites.append(Site(path, node.lineno, kind, status, detail))

    def insert_namespace(node: ast.Attribute, home: Home) -> None:
        end = source.offset(node.end_lineno, node.end_col_offset)
        edits.append((end - len(node.attr), end - len(node.attr), f"{home.namespace}."))

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and is_package(node.value, aliases, plain):
            if node.attr in homes:
                insert_namespace(node, homes[node.attr])
                site(node, "attribute", "rewrite", node.attr)
            elif node.attr in REMOVED:
                site(node, "removed", "review", f"{node.attr}: {REMOVED[node.attr]}")
            elif node.attr == "__all__":
                parent_compare = False
                for compare in ast.walk(tree):
                    if (isinstance(compare, ast.Compare) and len(compare.comparators) == 1
                            and compare.comparators[0] is node and isinstance(compare.left, ast.Constant)
                            and compare.left.value in homes):
                        insert_namespace(node, homes[compare.left.value])
                        site(node, "attribute", "rewrite", f"{compare.left.value!r} in __all__")
                        parent_compare = True
                if not parent_compare:
                    site(node, "dynamic", "review", "reads the top-level __all__")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ATTRIBUTE_CALLS \
                and len(node.args) >= 2 and is_package(node.args[0], aliases, plain):
            name = node.args[1]
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                if name.value in homes:
                    start, end = source.span(node.args[0])
                    edits.append((end, end, f".{homes[name.value].namespace}"))
                    site(node, "attribute", "rewrite", f"{node.func.id}({name.value!r})")
                elif name.value in REMOVED:
                    site(node, "removed-name", "review", f"{node.func.id}({name.value!r}): {REMOVED[name.value]}")
            else:
                site(node, "dynamic", "review", f"{node.func.id} with a computed name")
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in (PACKAGE, f"{PACKAGE}.fdtd",
                                                                                         f"{PACKAGE}.fdfd"):
            rewrite_import(node, source, homes, edits, site)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            dotted = re.fullmatch(rf"{re.escape(PACKAGE)}\.(\w+)((?:\.\w+)*)", value)
            if dotted and dotted.group(1) in homes:
                home = homes[dotted.group(1)]
                start, end = source.span(node)
                literal = text[start:end]
                old = f"{PACKAGE}.{dotted.group(1)}"
                if literal.count(old) == 1:
                    edits.append((start, end, literal.replace(old, f"{PACKAGE}.{home.namespace}.{dotted.group(1)}")))
                    site(node, "string", "rewrite", value)
                else:
                    site(node, "string", "review", value)
            else:
                for match in re.finditer(r"(?<![\w.])(?:mw|maxwell)\.(\w+)", value):
                    if match.group(1) in homes or match.group(1) in REMOVED:
                        site(node, "string", "review", f"mw.{match.group(1)} inside a string")
                        break
    return apply_edits(text, edits), sites


def rewrite_import(node: ast.ImportFrom, source: Source, homes, edits, site) -> None:
    stay, moved = [], defaultdict(list)
    for alias in node.names:
        if node.module == PACKAGE and alias.name in homes:
            home = homes[alias.name]
            if not home.importable:
                site(node, "import", "review", f"{alias.name}: import mw.{home.namespace} and read "
                                                f"mw.{home.namespace}.{alias.name}")
                stay.append(alias)
                continue
            moved[home.module].append(alias)
        elif (node.module, alias.name) in BACKENDS:
            moved[BACKENDS[(node.module, alias.name)]].append(alias)
        else:
            if node.module == PACKAGE and alias.name in REMOVED:
                site(node, "removed", "review", f"{alias.name}: {REMOVED[alias.name]}")
            if alias.name == "*":
                site(node, "dynamic", "review", "star import")
            stay.append(alias)
    if not moved:
        return
    start, end = source.span(node)
    line_start = source.starts[node.lineno - 1]
    indent = source.text[line_start:start]
    if indent.strip():
        site(node, "import", "review", "import shares its line with other code")
        return
    statements = ([import_statement(node.module, stay)] if stay else []) + [
        import_statement(module, aliases) for module, aliases in sorted(moved.items())]
    edits.append((start, end, ("\n" + indent).join(statements)))
    site(node, "import", "rewrite", ", ".join(alias.name for group in moved.values() for alias in group))


# --- markdown ---------------------------------------------------------------------------
def _dedent(body: str) -> tuple[str, str]:
    lines = [line for line in body.splitlines() if line.strip()]
    indent = min((len(line) - len(line.lstrip()) for line in lines), default=0)
    prefix = lines[0][:indent] if lines else ""
    return "".join(line[indent:] if line.strip() else line for line in body.splitlines(keepends=True)), prefix


def rewrite_markdown(text: str, path: str, homes: dict[str, Home]) -> tuple[str, list[Site]]:
    sites, edits, fences = [], [], []
    prose = re.compile(r"(?<![\w.])((?:mw|maxwell)\.)(" + "|".join(sorted(homes, key=len, reverse=True)) + r")\b")
    for match in FENCE.finditer(text):
        fences.append((match.start(), match.end()))
        if match.group("lang").lower() not in ("python", "py", ""):
            continue
        body = match.group("body")
        code, prefix = _dedent(body)
        line = text.count("\n", 0, match.start("body")) + 1
        try:
            new, block_sites = rewrite_python(code, path, homes)
        except SyntaxError:
            if prose.search(body):
                sites.append(Site(path, line, "markdown", "review", "fenced block does not parse"))
            continue
        for item in block_sites:
            item.line += line - 1
        sites.extend(block_sites)
        if new != code:
            new = "".join(prefix + item if item.strip() else item for item in new.splitlines(keepends=True))
            edits.append((match.start("body"), match.end("body"), new))
    for match in prose.finditer(text):
        if any(start <= match.start() < end for start, end in fences):
            continue
        edits.append((match.start(2), match.start(2), homes[match.group(2)].namespace + "."))
        sites.append(Site(path, text.count("\n", 0, match.start()) + 1, "markdown", "rewrite", match.group(2)))
    return apply_edits(text, edits), sites


# --- driver -----------------------------------------------------------------------------
def targets(paths: list[str]) -> list[Path]:
    inside, outside = [], []
    for item in paths or SCOPE:
        resolved = (ROOT / item).resolve() if not Path(item).is_absolute() else Path(item).resolve()
        (inside if resolved.is_relative_to(ROOT) else outside).append(resolved)
    files = []
    if inside:
        listed = subprocess.run(["git", "ls-files", "--", *(str(p.relative_to(ROOT)) for p in inside)], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.split()
        explicit = any(EXPLICIT.search(str(p.relative_to(ROOT)) + "/") for p in inside)
        for item in listed:
            if not item.endswith((".py", ".md")):
                continue
            if EXCLUDE.search(item) and not (explicit and EXPLICIT.search(item)):
                continue
            files.append(ROOT / item)
    for root in outside:
        files.extend(sorted(p for p in ([root] if root.is_file() else root.rglob("*")) if p.suffix in (".py", ".md")))
    return files


def run(command: str, paths: list[str]) -> int:
    homes = load_homes()
    if command == "table":
        print_table(homes)
        return 0
    sites, changed = [], 0
    for file in targets(paths):
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if "maxwell" not in text:
            continue
        rel = str(file.relative_to(ROOT)) if file.is_relative_to(ROOT) else str(file)
        try:
            new, found = (rewrite_markdown if file.suffix == ".md" else rewrite_python)(text, rel, homes)
        except SyntaxError as error:
            sites.append(Site(rel, error.lineno or 1, "parse", "review", str(error.msg)))
            continue
        sites.extend(found)
        if command == "apply" and new != text:
            file.write_text(new, encoding="utf-8")
            changed += 1
    counts = Counter((site.kind, site.status) for site in sites)
    if command == "classify":
        for (kind, status), count in sorted(counts.items()):
            print(f"{kind:10s} {status:8s} {count}")
        print(f"{len({site.path for site in sites if site.status == 'rewrite'})} file(s) to rewrite")
    review = [site for site in sites if site.status == "review"]
    for site in review:
        print(f"review {site.path}:{site.line} [{site.kind}] {site.detail}")
    if command == "apply":
        print(f"rewrote {changed} file(s), {sum(c for (k, s), c in counts.items() if s == 'rewrite')} site(s); "
              f"{len(review)} site(s) to review")
    if command == "check":
        left = [site for site in sites if site.status == "rewrite" or site.kind == "removed"]
        for site in left:
            print(f"old {site.path}:{site.line} [{site.kind}] {site.detail}")
        print(f"{len(left)} old-form site(s)")
        return 1 if left else 0
    return 0


def print_table(homes: dict[str, Home]) -> None:
    by_namespace = defaultdict(list)
    for name, home in homes.items():
        if name in OLD_TOP_LEVEL:  # names a parallel 0.7.0 branch added never were 0.6 top-level names
            by_namespace[home.namespace].append(name)
    print("| 0.6.0 | 0.7.0 |")
    print("|---|---|")
    for namespace in sorted(by_namespace):
        names = sorted(by_namespace[namespace], key=str.lower)
        print(f"| {', '.join(f'`mw.{name}`' for name in names)} | `mw.{namespace}.<name>` |")
    for name, reason in REMOVED.items():
        print(f"| `mw.{name}` | {reason} |")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=("classify", "apply", "check", "table"))
    parser.add_argument("paths", nargs="*", help="paths (default: the p5b scope of this checkout)")
    args = parser.parse_args(argv)
    return run(args.command, args.paths)


if __name__ == "__main__":
    sys.exit(main())
