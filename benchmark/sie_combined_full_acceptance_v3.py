"""Source-frozen single-device provider acceptance: collect, reviewed shards, aggregate.

Numerical source remains the reviewed 0c364a40 tree. Collection executes no tests.
Multi-device and pFFT coverage remain explicitly unqualified.
"""
from pathlib import Path
import argparse, ast, contextlib, hashlib, importlib.abc, importlib.machinery
import importlib.metadata, importlib.util, io, json, math, os, resource, subprocess
import signal, sys, tarfile, time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Timer
import xml.etree.ElementTree as ET

SCOPE = ['tests/sie/test_archive.py', 'tests/sie/test_block_gmres.py', 'tests/sie/test_bounded_validation.py', 'tests/sie/test_combined_direct_storage.py', 'tests/sie/test_compact_riesz.py', 'tests/sie/test_cross_assessment.py', 'tests/sie/test_cross_backend.py', 'tests/sie/test_direct_restore.py', 'tests/sie/test_fields.py', 'tests/sie/test_frequency_mesh.py', 'tests/sie/test_linalg.py', 'tests/sie/test_matrix_free.py', 'tests/sie/test_mesh.py', 'tests/sie/test_modal_reference.py', 'tests/sie/test_modes.py', 'tests/sie/test_numerical_audit.py', 'tests/sie/test_observables.py', 'tests/sie/test_operator.py', 'tests/sie/test_physics.py', 'tests/sie/test_public.py', 'tests/sie/test_quadrature.py', 'tests/sie/test_reference.py', 'tests/sie/test_single_buffer_assembly.py', 'tests/sie/test_singular_convergence.py', 'tests/sie/test_transmission.py', 'tests/sie/test_volume_reference.py', 'tests/api/public/test_public_api.py', 'tests/sie/test_fmm_config.py', 'tests/sie/test_fmm_scalar_near_config.py', 'tests/sie/test_right_preconditioner_hook.py']
BASE_COMMIT = '5093aa2032360ae353b0d034ee754230ce1da93e'
SOURCE_COMMIT = '0c364a40c3a6888f661aa9ea50d355b81b908a28'
SIE_TREE = '6f739365f561f35b57d6380e928690666eba577e'
PROTOCOL = 'combined-assembly-direct-storage-v3-dense-core-defaults-ac'
PYTHON = os.environ.get('WITWIN_MAXWELL_PYTHON', sys.executable)
BINDING_HELPER_SHA256 = 'f996a2d65ecddec6d25c42c1ea195fa8bc14dcbab6e3dfbd4711ad2675b69725'
SOURCE_SUFFIXES = {'.py', '.cpp', '.cu', '.cuh', '.h', '.so'}
MULTI_DEVICE_NODES = (
    'tests/sie/test_numerical_audit.py::test_gauss_cache_resolves_implicit_cuda_device_before_lookup',
    'tests/sie/test_successor_robustness.py::test_actual_compact_dense_solver_matrix_and_cross_device[dtype0]',
    'tests/sie/test_successor_robustness.py::test_actual_compact_dense_solver_matrix_and_cross_device[dtype1]',
)
ARTIFACT_UNAVAILABLE_NODES = {
    'tests/sie/test_fmm_scalar_near_config.py::test_all_old_payload_and_construction_bytes_and_strict_scalar_dk_versions': {
        'reason': 'artifact-unavailable',
        'qualification': 'unqualified',
        'detail': 'Original configuration source pair unavailable; explicitly excluded by the user. No artifact supplied or regenerated.',
    },
}
UNQUALIFIED_FAMILIES = {
    'H/H2/direct-H2': 'Source identity only; no hierarchical backend qualification.',
    'FMM': 'Source identity and explicitly named default/hook controls only; no general backend qualification.',
    'tensor': 'Source identity only; tensor workflows are outside the dense-core scope.',
    'periodic': 'Source identity only; no periodic workflow qualification.',
    'volume/harmonic/static': 'Source identity only; no coupled volume, harmonic or static qualification.',
    'open_surfaces': 'Source identity only; no open-surface or EFIE qualification.',
    'pFFT': 'Source identity only; no precorrected-FFT numerical qualification.',
    'near_field_LU': 'Source identity only; no sparse direct factorization qualification.',
}
REVIEWED_DELTAS = {
    "__init__.py": "e402bba5110e7fa74acc3dc81571fc7153b6126796ff65b2c5d7a616f64063c8",
    "_fixed_inputs.py": "44389425329906fd1f4ad5ae5d02697134473efeecaf88bf9f41ec2ec429bf53",
    "absorbers.py": "edbc50b993e570c1c6ac54c63608850acac3671a42898ad8a945cc966d0e48dc",
    "adapters/__init__.py": "2f07fba2a2ab24e13ae32e446c7f84b35aaef7adc3f34668d354e15e542ff746",
    "adapters/pcb.py": "20a8fe93f7718a38bd66180892732d8cb3d065666ba37cd192943a9fbf641270",
    "adapters/tidy3d.py": "0902fb690038a526f250408e6b5d23f8da0471d89216d3104a5d3a29726f1791",
    "adjoint_inputs.py": "1874e4e5789c0fae380a124cd1f6aa1947643cf22b22c549e05ce2f1b92fc7fd",
    "compiler/_scalar_cache.py": "290ad2b618f0cd17196f2eae7f8b2016e49744e62f0a34c3f019b1f895139af8",
    "compiler/coaxial.py": "738b1c81e91496e5fd5c89d40481bde382e8deb42c4436876370a43edd3713d1",
    "compiler/coordinates.py": "946a5b3d24c83f10dd5fb5dfd0947e64e2f1b9655adf971d39bbd23bcb852ec0",
    "compiler/copper_foil.py": "c227a35270a1cc357566843da3d17fe752a3f79216a9fc24259197c09893f362",
    "compiler/curved_copper.py": "be7b77ed59fde79fdc0e070106890eb855f20f8d1821f88febae63cf01c3980b",
    "compiler/cut_surface.py": "15ec0e84aad722cb58f305d2c483d39f9bd196c06710ce68483a516c5530dded",
    "compiler/electrostatic.py": "1c06442b6cf8815d2fd5e1fb4cbba9a900ec90832040e3881225b718fe9cea2a",
    "compiler/fdtd_audit.py": "708a13446a3365f0bcf41e193b096a59047c51923c43ad466f018d65a609ade8",
    "compiler/lumped.py": "759bb018089c06adbe64abf82a943e533efcf37d2f9db61f12c37e5bd5ba962a",
    "compiler/materials.py": "a8cac097826d040dc289b4f71043a10dbae78e30ff370a22e8782ab7447e8651",
    "compiler/monitors.py": "22675b04204c974a91003b3e2502895d6519e58306f8a67dd4d307fa8385aa9b",
    "compiler/path_integrals.py": "72ee8aee7a83e4b3ed02cd38cc6dd4e6274e67e23879674690d9d785e2842d9e",
    "compiler/planar_pec.py": "d755515f9d686ebb0b055074dbe03c2efc70baa4b2f8046815bc32f7a105873c",
    "compiler/ports.py": "8293eab2bcc483b7249b1e71760f47c00fdf987f7d849860a9f2dda516b802f6",
    "compiler/power_loss.py": "4c174a900fd3dba139cfff621b4cd0de4a28bf2392455b59b286c5714fdf52f3",
    "compiler/sources.py": "ae8275c9e584087873ac632b2fdcc281a30cc75b1e1cf1d327c1854ac38961e6",
    "compiler/structures.py": "cb446ad9f181033cee46a8ad426f77febdda3f7b61e030e227ecef408a641a57",
    "compiler/subpixel.py": "ec62ebbc66489f7d94e3864d589b92519dae57db61ba41cd8530d449b4e35227",
    "compiler/terminal_waveports.py": "8856d052164ccb510da0efbd127d0af07ae23e796c94ce1e1e4d33009198dcab",
    "compiler/waveports.py": "c0a2977a6c355223e4eb7dd58a92a0f8b3299f28a2f3470e729ea2f4384d8bcc",
    "coordinates.py": "a647a76245ea77049554de592897666b08f95cbd75e1dba63d4524d41f270efc",
    "copper_foil.py": "b7081e2579a9f5750e8e5e5191abd28a94b719fbd8f24d97069ac194c40edd12",
    "dispersion_fit.py": "9011855279778a00a246a6d5d0432adab9a732ba74cbce16776e4da9edbe2d39",
    "execution/adjoint.py": "9dc6d04cbae1df2a7c357c27cc32666582449b3b16f3fb882aabedeebe40afd5",
    "execution/batch_runtime.py": "72510195b48eec526a880b0febb48fb50abe7548ac8c528f5a869d99c7abdd30",
    "execution/plan.py": "6fa6436914a594788a9c7f34b1a6462a238b8ab67bc4394373d0c57fdea0a30a",
    "fdfd/adjoint/bridge.py": "bca7f4b77bdaf920449ad509851dfd1f1a4362ca28f2363c018764750d859535",
    "fdfd/ams.py": "99dee3f2f1ad6be75c6f1d64018e11ebe44f745eb7657b6c030f186f278426f2",
    "fdfd/magnetic.py": "3003212b140b868ef71b82ea8462f707c7d6976d086f65a7409cd27d1232df8e",
    "fdfd/materials.py": "25dd87ed2259c97fb3125826b490953f3502fa1a8a055def8fd485c2b56f59b3",
    "fdfd/modes.py": "5c344e24f1d5c30a42005612febb8fbae64147045b92a38e772b631e22d2daac",
    "fdfd/observations.py": "8e4cc99fb690a92c824ff7eea6cd88dc26672385a11b9b9b7cd5242c3ce78c19",
    "fdfd/optimization.py": "f1ce299a0f8554c539481a17f857e8e35738eee5c8763379a096e8ac24585e4e",
    "fdfd/pml.py": "41a23d41222ee199c2c6ae094f51e84545e24a8f5aec37bc8ff4025f27771248",
    "fdfd/postprocess.py": "f26a4d70307e936948065dc0c85bfaef45cd4a98c8dff9bbef45fc5cfde9c220",
    "fdfd/response.py": "2711e5b2df3ac10ab37601aff74db2ab8fa10072b1076019c54f9b60e4160ca5",
    "fdfd/response_cache.py": "d297bbd349cccdf84ecb3b6f50797b07a82a1e500ce6cc9787739900ee49a7c2",
    "fdfd/solver.py": "0c2abd6d5d4f528135d9a4dfbfd93642f29978ab4a03e9a5c272530a133ecc1f",
    "fdfd/sweep.py": "af6d55f55e814b51a3b118edd08a6b315e2481f0e36de2dd2fc2bc51e5a61449",
    "fdfd/update_cuda.py": "77f8a086d20a945d196ff0f1172b8c6222908bd70b10f82a03e0725d25582636",
    "fdfd/update_solve.py": "4c6220727c846ddb2ab3d2d14a649ae5154563e6310913b32eccc2303f9d13e5",
    "fdfd_response.py": "824cfaaefb29ee68209f254a410a148fac8a70a0f35bf773afaa538ef6489f1b",
    "fdtd/adjoint/bridge.py": "e277b5b2eb28a93470961465620cffdd6d46e3271169ebfd7a4b3caadd0cf17c",
    "fdtd/adjoint/capabilities.py": "43e087a2fde65b16e67ed92a42e2f403dc884b07c99839e2a0cb52f744e4c3d4",
    "fdtd/adjoint/circuits.py": "5d043bf5f9c6342b65a33e1d07fce77781db8eb9c8b63be99dc7a96238595f28",
    "fdtd/adjoint/copper_foil.py": "1e2e1ab10ad2ce8800ef66e6f4bcccfb20b55c52f47cc6a5660921146e4e0ed1",
    "fdtd/adjoint/copper_spectra.py": "b3b478fb7b6be68ffdf65ec3315c13670deaa71b36f8fbfb10681067ea1fb34e",
    "fdtd/adjoint/core.py": "0c7bc0daf3ae22b5698a7c086ebf001c494ebb147ca5fc9e70c1a15098c1fc18",
    "fdtd/adjoint/dispatch.py": "412094dde0bc3f7411109ba541166afe56374b6521e11658b7a53f628a74bbab",
    "fdtd/adjoint/joint_material.py": "10daa28cdd99b9a2bf2cdf3913b050abf54e9d4f480b5f9657a63b2a8fea0a50",
    "fdtd/adjoint/mapped_admission.py": "796b666089c4bc8b364c84a6b3c7fb4550f0a1f7697ee9fee35988fd42e33f54",
    "fdtd/adjoint/native.py": "8b33126cbd98d142e8bdc1fd0466ff8c6b13e3a2440e5280397206fc7f34cca9",
    "fdtd/adjoint/profiler.py": "c0372911053384390530576c9883fd899067d18799dcbcb4534ed9c81d67bf0e",
    "fdtd/adjoint/seeds.py": "8fb99302f9dd7ca00f9ec45f554d93e14d3b64652eb7fd5dda0f7a7c8ec0693e",
    "fdtd/boundary/abc.py": "9f170531776cc9438eb3230f7fc0024cfcaa02ed3a26d1433cab7155ff099c44",
    "fdtd/boundary/common.py": "e80f88052cd3ec416574dc6a17b2dfeb7bbe35580f384753c28b595ff1e01353",
    "fdtd/boundary/cpml.py": "742059637dfa510f69674fad19ce695342388865735c9941e8f20f4c487d571f",
    "fdtd/boundary/mur.py": "d403da5adc3946e9ed776283d3dc61fcb12f63409bc07d61ab8bfbe07b4fabce",
    "fdtd/boundary/runtime.py": "4ead5bb5ab87229fef3243723bc415e9284e30fd46a6d17a48be4126c8cb7e82",
    "fdtd/checkpoint.py": "97fdaf99752abec94beb69cb4e3ee7dd3b44e6b960ff74d0ac07b3944e424ce2",
    "fdtd/circuits.py": "270be3303b6b4b8b3f9a6d15788c7a8dafbcdb0d0407b5b65f7becc1b5cd9b2c",
    "fdtd/coaxial.py": "eaedf71e299e946397fa8a0cd387d99b31d82976e489407dddd8241f9753cbd9",
    "fdtd/copper_foil.py": "62acbfabb271a901353ad320e37a884f775e991cdbceb2ed7ff7a50c12177db1",
    "fdtd/copper_foil_ade.py": "2fae79ebd91103ff37a616da39b1f8a48a6e163f6478740df9b67d2221193fda",
    "fdtd/cuda/backend.py": "6de151185de466440081c5066af3c5ae87919f62c542f0fa7eabb97ba4a66486",
    "fdtd/cuda/build.py": "3e4b85f94fd960dd6ca6d35dbb1a9b0153cadb192fcc458612bbb6c62bfec140",
    "fdtd/cuda/extension.cpp": "6dc5b6eeba2aba9958852cebf9adbbc3d94a7dcdfd6698cf8b049a8d8ce690fc",
    "fdtd/cuda/kernels/electric.cu": "f80c6bbbb71190c7fed10e130933ef9ed73f929a1ddccf5e964b94175d4aaf09",
    "fdtd/cuda/kernels/magnetic.cu": "f9da00820ba9a1b0f1a95c1034a10142059ec761772af9e14afac85602794664",
    "fdtd/cuda/kernels/observers.cu": "11f9d8c760342bfa15b13bf1f107f8b87cc573f5886cd765c04259a71f58d852",
    "fdtd/cuda/kernels/sources.cu": "e660d63129cc57a1195635d6024cba04c3e71376f561765532799e3fa3ceaf3c",
    "fdtd/curved_copper.py": "2e0870f3a6702dff236c9493c01eddd5d72c98d414dfe9e541475daaf3a523b9",
    "fdtd/cut_surface.py": "3b2214b6b111d812b8bdf3741154b3000a06c987928b41c384019a6b58cab0a8",
    "fdtd/distributed/circuits.py": "94a717e08c1b5165b65089a5986847f4a634f49ff4286ce310f648e6f92100b5",
    "fdtd/distributed/copper.py": "aa7e071bf452e75892c7035b540e15127fe223f439b6a7ceca9041a4bebd1bf8",
    "fdtd/distributed/monitor_merge.py": "aa5072d6b67eb94b1f6067754fc527ce6ca080cc311184a72a604f47738a6acb",
    "fdtd/distributed/networks.py": "2b0e8aecb44167501300431617900803b036c1a135233ab43d3190aa603195c5",
    "fdtd/distributed/observations.py": "0ed355f1b2114e933e56a60645319edee416baabd03fd91f83a3dcc5b0962b92",
    "fdtd/distributed/resume.py": "c553355efdb876483baf8563bad41ca1000cee079ec67dadc97c6d5cd86af93e",
    "fdtd/distributed/shard_engine.py": "0b17c3534f4c7aed6da6569a65c45161da8e2f111805c11d37a9b184507f89d7",
    "fdtd/distributed/solver.py": "50f3c8f15a365c182c059c304763fc5062124946c1360b747eed891077978d4b",
    "fdtd/distributed/source_rebuild.py": "8d69ce7b7915dea4ab2d2f72830ab6b1cdbb84121be5730f48f4911a4034904c",
    "fdtd/excitation/injection.py": "04497e1125027d9cb52f42a4416816f5784b1dd26920d20154adfc169af80c74",
    "fdtd/excitation/modes/__init__.py": "5c9cf52e21869f17d7f809f5aae4d0254f54612b54fc914046273851dc267bac",
    "fdtd/excitation/modes/common.py": "2b793ba75383c7a6ccac73693c332ac9d99d4a073a771f05062f9f6f62368e98",
    "fdtd/excitation/modes/conductors.py": "3b9de705a429edb8557471a17d54f80828a745f605fc24c4e62f64115bb869f3",
    "fdtd/excitation/modes/tem.py": "f4419f020effc30448139a54f2f0652b20f917d94cfc0fe1b5987f2fe3cd08b4",
    "fdtd/excitation/temporal.py": "959759d5ab349f9b708b45631e4be69c03d4a30debe94244dc5cc8de21bb275b",
    "fdtd/excitation/tfsf_common.py": "9c2eb21da33270711de179eca0be4fcfad73bbf47ba8339966a40b408f777e82",
    "fdtd/excitation/tfsf_state.py": "1d8b827d381900d90c5ceb6c369cebd07ddea458c877629a863a7a77a2289342",
    "fdtd/lumped.py": "61bce3c3726c433972c4b3845dec1be95bdc0905040e17c6b28e408454ae58f2",
    "fdtd/material_pullback.py": "b38b380d485243c3087d91ccebb952f266f39b3883712b3adf1a221f86c68443",
    "fdtd/meshing/autogrid.py": "98e03ccd5e6deb7cf47d4a40209d130a84eb5897b2f655bda7ee09051186afa4",
    "fdtd/meshing/corners.py": "7f038bff5c8fd917523b00214fa5ecd1b99a14a7b760a9048a7bbb578920ab61",
    "fdtd/networks.py": "a94277f53952ebf95470be71e2c5ea388f35f3d47480223b25527b8dbe5200c0",
    "fdtd/observation_resume.py": "267c3b1c20f78aa1143f590cd64beaaaa2443d6f45c9bcf2d01328308af80dcd",
    "fdtd/observers.py": "97adba54c2c8a95e2dfcab9a94bb37a18deb6ccde731a0df2d3797da99f4b3e6",
    "fdtd/plane_product.py": "1612d2d56410b667de667cf923a3d7b8bdfe67de3cbe5b37f1032acfa3a49cd5",
    "fdtd/ports.py": "b09c1b5a2ab92d4e9b39a6a494ef75e58e816268eb331c9f8e3239b589e8d7a8",
    "fdtd/postprocess.py": "955024f15f59538a91b02f8ab9a9efb657a66cb220da3a143349ca460dcc1ff8",
    "fdtd/quadrature.py": "e3ac718e3bba09127a0592ea09e4fb1ab4efa429f37e916c18ba528098f2c389",
    "fdtd/resume.py": "eb403e92dd4f3ec7b92797299b2c9aeb4172687638359c71baf48a096646ad84",
    "fdtd/runtime/batch.py": "f24f39a03093e88d3f5b2b063a6b449cb9e2b999b65cad7e8fbb3d8146c66210",
    "fdtd/runtime/batch_materials.py": "ba3d4e395f32b5967d531c260b704209de90170f06871f38a7249f4f04062d73",
    "fdtd/runtime/initialization.py": "0beb8ac2a76afafd6a395f1195f40243e4f59355d4c148cb7b2cee98a340b573",
    "fdtd/runtime/materials.py": "4d6926c459c63deba32f20faeb4154292a809a0bd88c88eb9593bf0ea19daf1d",
    "fdtd/runtime/precision.py": "15724f63c27abc0f8c68fb8e7d16685ccef1c6a96ed02ca683a51a18d94b827c",
    "fdtd/runtime/representation.py": "d453ccca3622bab2a6ab7e19f57679d770fc73d0c59bdc51b42480ff3e3d3136",
    "fdtd/runtime/spectral.py": "1b64f6636b820c1565141fde30f4a231b9101433b71bcfe6214580ed209c3f96",
    "fdtd/runtime/stepping.py": "df506c6943406d11ec928d8be05b7fa6efce05fd770977afe49cda8fa3e75296",
    "fdtd/runtime/timeplan.py": "f6c1861b40659154d5835be6fa28603356bb84bcc4f3bf15aee9643e03d133ab",
    "fdtd/time_flux.py": "f0ead223980e4bbb766378132615835f40a97ecf1be600ee995f55d1384a9d31",
    "fdtd/time_traces.py": "5e5e12328c588946f242665dc58e8f5858813041542f18024c3b2de6ccbc3a4f",
    "geometry/__init__.py": "5f9d038828708cb4d722b5857687f2df5f982d9918e7c3def86d8117f07b674f",
    "geometry/planar_pec.py": "b1afbfa3ce166f3b3d8d356afb1f8a762ecc2bb7c5d91e375e09a89446b584fc",
    "geometry/surface_mesh.py": "d526b36c5f3038f2afd0612a6b5d6456d036facf677dde7553b09924f6f65d72",
    "geometry/terminal_support.py": "cbf7f41aaa623e00cb80cdd4953c043eceb9a37f71b1d979bf9feed129f2d474",
    "invdes/__init__.py": "90fc8c2113368d8ee73138c2de30ffcf2077e292e69bc728ae0102c77b5ec8ab",
    "invdes/constraints.py": "801af1746e6c22d59ed180eb4fe930114eedb01b42cc20a9dfa9b1698177bbd6",
    "invdes/filters.py": "170e9774664261c9d0b29116b81e85571a7f219d74c5d5b04d45cbd5ea013c4b",
    "invdes/optimize.py": "7347a55a6158953fa6f4c49aa796b77a9ac607360925003c8b5c807e8990f186",
    "invdes/parameterization.py": "42d777b43266116b5919bbab51dbc8d9aadb28e67fc5e35b2d2ba596754e41d0",
    "invdes/projection.py": "fc927cc5d797980911ec92804f055b5eedfcc95a57709b559f27a8869c9e013c",
    "invdes/symmetry.py": "8e09fa0f5e7236baef26c01905960a114da05a8dcddd3b997ef4615560ffb962",
    "lumped.py": "b73cc38cc18f91e44d4c4143f4071c8c6c55df9ed866e819515e6e9d4a5317f3",
    "material_cache.py": "a3ccd3ecead35e65f1330c37b3f798ff96ede214d999a63d09f1a5074d3069df",
    "material_library.py": "45a222dcbc02d4ab213cf3c2ded8f902391ae5ab026bf6047a87215caa60bfa6",
    "media.py": "4ae969ea16a31e9f55436d8c519b4002dff13abafed2ae52a0ef497f2c9ac021",
    "modal_sweep.py": "eec957da36260fbf3bff5f256c4b61553b442e3e90311898dd6cb87c07f638e9",
    "monitors.py": "77b4b079a3ce6f8ef3954166dc250ac5dfb473306f6342b95cbea0fd4741ae88",
    "network.py": "8616e6aa1361a53207372d999e05db2b987d2e7d3f95f69f36ddfb71206b73e7",
    "network_math.py": "c90ee7782a6c48350d433139c98430439295ce77d8f5a24707f0e8af51d522fa",
    "path_integrals.py": "cb8e448b109b6b8173da26408bd839caf8f6389677c96568961fd3459cca50cb",
    "ports.py": "54e1686ae27c5084d02744d2563eaed294e23f07400133c069acffcbf0a1ec55",
    "postprocess/RCS.py": "432fb7223b5f461918880ae3f90ef22574ab3faa6c1d6bf040c6ba777de5d37e",
    "postprocess/__init__.py": "f8ddf3f37753c1c22f0a19f98f587c5869ee8e654943704aba6e4d11749a90e9",
    "postprocess/antenna.py": "f3243d1e95f32b38f52c916a763dc430d5d940acdfaf649bbd73a4f7a68a0cf6",
    "postprocess/array_factor.py": "828dff410d7f91086033e71a3e020798fc211797e88e3a6561d7e427b9cda9e0",
    "postprocess/array_taper.py": "25b7a4124cabd0d1972a8945f4f998dfe4d18359a993ea8f9d7137dc899f6329",
    "postprocess/fdtd_loss.py": "9feff8237480f1a31a0896e101c4d053bff798c0ec6bb64c021dfa7374db68f9",
    "postprocess/field_projection.py": "2f77caf627bf08644669fdc615166f0cc0c78237166a5f0e7b8337268dfcc3af",
    "postprocess/gaussian_overlap.py": "a59c4a1f16ac4401680a241ea86b8ae611752824dd87efe593cf609f0e1c799e",
    "postprocess/lobes.py": "28848f0ea4be6f2c7985143acba55e081b9b2cfb716ec518a85e3ad6f8c2aa38",
    "postprocess/low_frequency.py": "195ec5e08c4c2d0633b01fde03f6361ee3b05707b4b92f28f12a5fd6a0408f57",
    "postprocess/modal.py": "ec406ae4d043294d5069ad72282aac6387616c6fc6036d0333a10e930c080cfc",
    "postprocess/resonance.py": "a5074d56038a60d0aa1b9aab943e7edbf9e1443d4d71c14f534f1dd84e1fd446",
    "postprocess/selected_query.py": "3164e0f4d0a8ebdbe90b2bb53fe1ab6991dbb827d0ddeead3543954d61ff054e",
    "postprocess/stratton_chu.py": "ad2e75ceade8502faa78b63651c5ba7aebe9f91b1399f01949ab81eb3f6e6a01",
    "postprocess/yee_interpolation.py": "f8144855faaff8b7b89835b1c563662e34ddf22053da85d97d081f5c4e3f5eb5",
    "power_loss.py": "58a2c6d1b4a87e478f20996910d1011f2a642f50b2586f8db044773e26491cbf",
    "result.py": "b4ee3b33bfccf27ddce28f2f4b2694c958ef097fa73e28e30703f1ac56d35420",
    "roughness.py": "59409a3c31a796cae42949b04a80192d9a5d52d6de807a5c969b9132eec9ddeb",
    "scene.py": "671d53628ef5ffb164528880014d1584d916e9d50dadd478fbcef423c6461e68",
    "sie/action_harmonic_balance.py": "d47436e867bfc2648ebf09e65211d6790db7095a0247db450585bd21a0d08fd4",
    "sie/anisotropic_fields.py": "f41c76384c1b4d23790cfb7f149d271fe852af0deb8da82b687f04610375c9c2",
    "sie/anisotropic_lifting.py": "84e46e9fb6b9e71f89c00b9da78bfc5c479908b25702d8dd5362ffaeee61a7c5",
    "sie/anisotropic_operator.py": "9f365164ef0d3b7952ff59be6a818896da4e3fca5974b3528cadfd5924ef1e0e",
    "sie/background_bank_snapshot.py": "7f328b0b275e26cb5fda5071c0add3ad0c2bc22cb143b1d04d12413e5674fcd8",
    "sie/block_preconditioner.py": "5edb1172af0f6d9128fb11fbaf464a34b12ef6bd2831ed40e59d32dc23eb0a4c",
    "sie/carrier.py": "fb2868a20beb560aed6a464d49e0c866d055a97881f25a92244e1a71dbb59f0a",
    "sie/cluster.py": "d3d23fed04fc5abd92cf8fcf07a4fad770eaf0762282f3c6a2e6291b1c960978",
    "sie/complex_static_tensor_green.py": "ddd3d945b8f2ebaf90367a5e1e4bce4cc764fb84fe4ba99c9f9d36442b82368e",
    "sie/compressed_background.py": "b0d5e1167dbdd7cc2a273ded5f8f2ce709b095125a07e52d6b61ced29f8adb4b",
    "sie/compressed_harmonic_snapshot.py": "fa722eb68fe15eed34256f7c6122234076a7a68054e2cd7ad010a840669d7074",
    "sie/compressed_volume_preparation.py": "d811347158168972f5c0752e3b3b7ab74062c522b5fa235051cba77f42297060",
    "sie/conductive_static.py": "9a7fb9efd1cd1b17799a890569f96707f90f79681ae516513c28eba624c8b7a8",
    "sie/coupled_field_refinement.py": "7293e1a7f000e1caf5fc9e5183b06509a80733145214142be2f27c9500f34ceb",
    "sie/coupled_original_check.py": "9190bb45bfd21fa2d5fc5791d39dfe9655d516d60248f424fcafee385ce1f56a",
    "sie/coupled_response_snapshot.py": "e3603d6720d1f0e75b01944c0ba21a3d3a0085ed36b050857b9451e2bef6bb30",
    "sie/coupled_volume_action.py": "07adaaac3fedd2c1721cf8f1aa19dd5d5c0d7be3554d5ae4817a8679fdfb5a80",
    "sie/dipole.py": "4ea7c19f72e23660d28aabcc1a86be39a1eb136c3b9537f4f85092a1f91dadc9",
    "sie/direct_capacity.py": "25b322d14d065c2e46d9353ff785f2522d55f2d884bb3980ec7e23a25bf9debb",
    "sie/direct_config.py": "01e1fbaf0ff108b4a46ee92e3d980276bae4b94a3951acfc41277faa41d10f6a",
    "sie/direct_operator.py": "7029122804c66ded0780424e5dbba440e632d2a52c8c604bc824054cbc68c381",
    "sie/direct_regional.py": "fcaaf3da898c95f36164f8fa31479ab1b24316698e4d82b56f151753ae794751",
    "sie/direct_system.py": "17f1b07f20c22bee0877f998375b93db644010aee86337e9edb6878867fa770d",
    "sie/dual_action_snapshot.py": "3fd50ed0e392d58c2a789e3ee655def880be1ebc2630eb61c2bc86e12e27853b",
    "sie/dual_current_action.py": "c0a9883012e7f68a0ca0cfa32a4e512b59d9f9e6eb1c4c8798242ad156276063",
    "sie/dual_current_material.py": "a3f3544eaf902e26ce4e0f9299fd83144e03c2be1f13af1c3c8f4c982ca1f06c",
    "sie/dual_current_receipt.py": "ebcd34534d11046fa77a0448711b020e2d0a047042ba5e525f168082bcd42ebb",
    "sie/dual_current_response.py": "724b531aa5687422e175b5bccf4de49986b4db8eee4dadc3706d3eedcad94359",
    "sie/dual_preparation_snapshot.py": "f9778895fe10cc97fd88883d9690f7d035aa5cd9e91e35ba7d0a479affaa0c64",
    "sie/dual_static_current.py": "328d899edd0b257b110193e0669cfd360e36bc34ea1ce5acc44201a31b03bfe8",
    "sie/dual_static_snapshot.py": "5b7e3e7c05b8e74a34da062809eb673a118617c7a8e7301b5c885720779d3638",
    "sie/dyadic_contrast.py": "c23a68e740aa482f7929283e0834ba9972b859867d655746843288ef29c96011",
    "sie/dyadic_green.py": "763916b6fca9ae67e5e960dae2fbd12882c75620dad2e40099317cd6a96da47c",
    "sie/efie_open.py": "824e256c846d4e8e853915a70069d12fb2a3369d8e3d5837a1a7955a6f6b3145",
    "sie/efie_preconditioning.py": "1de13fe2b90efbf90b565740e89d576c1703d1706045a93a582358c1c3014865",
    "sie/ewald_nearfield.py": "7f228d14d09a3fdb991d8709c03d986b093bd82d44cd2891a6aad024b450029b",
    "sie/ewald_refinement.py": "1fbb136e24e91be3a4c798266e1746d16c3a90519b2d78f53a4433d37fc53a3d",
    "sie/fields.py": "c11e6a74aaafdacffbf2cde18d41e14f0e2dcac385fd3d7d8056e0a27542ee4f",
    "sie/fixed_tangent_inverse.py": "fc27c8740de227f6794353fcf5c2178ed1dfd48c8eaae005180b26057975934d",
    "sie/fmm_capacity.py": "0428de2dbe786a392f68f6508e43c817681792a41798a73e1aa539ee8eb83312",
    "sie/fmm_compact.py": "61fe5bd320c66dc54b261168d2a7e5e494e8b0734161f793f429641da6972d23",
    "sie/fmm_config.py": "7e86c78fa4fb5740fa52788a49628631cfea8809ce2c2ad079cda375a4ba1378",
    "sie/fmm_directional_fft.py": "c61ce72f4add30339709e5abaf18d2619ff99f083e7da2b064f8506e0431c37c",
    "sie/fmm_directional_moments.py": "ef54ff7e7dbf3ce1158e283cca3fbc8d741b14817c789598ec5551d0219a1a67",
    "sie/fmm_directional_rwg.py": "727347db5092dfb0b6c157b85ea01b9e5c4bd7139060a0481b24c3917e5e5b67",
    "sie/fmm_directional_shift.py": "c9595fcff015f5acac4344178b443a9d82a8df0c1ef357555ccce26017c46a4c",
    "sie/fmm_error.py": "75f169c7062055f9902d0e0e8c706ac11fe96c3996c909c4748e868c14a3a226",
    "sie/fmm_ewald_fft.py": "fc64f03b16f7ec4396b0d10908d74dab0ba62334857ea8fa392fdfe807150317",
    "sie/fmm_ewald_real.py": "009d6084743e091cc04797b72045f6292de3cf37442976c1baf663c8736b862e",
    "sie/fmm_ewald_spectral.py": "03b2b3e658771dbdeff97bca3a9425085a0bc39e6500f54f2c6a2e1750b669ce",
    "sie/fmm_ewald_translation.py": "12d218c64dd344138bbf0a1ba28b6c2e1124bcd964b959ef947eac063411810a",
    "sie/fmm_expansion.py": "082b2d06747dd370606d6811e746dccdc5103dd0ff6b70b0efb4c4c26bbe2abe",
    "sie/fmm_fft.py": "2c37c851169cf6d8133897d66b7429945465c13893a482d8323343f10b442068",
    "sie/fmm_fft_coefficients.py": "276174f85cac9c5a2db54ca67fa8967f3d52a4df1548019eccb99a510ad8ce75",
    "sie/fmm_moments.py": "557f6dfcd842f4d37e0535e4552e0a12f5f90d48b8905f4af6d49f51a12c6cef",
    "sie/fmm_operator.py": "c2403b1780e9d9476911ce03e8e1725ef710b2854096b6dd9b50de8179382dea",
    "sie/fmm_order.py": "a1caf9b63f80385879f9b6283717cd430511a3652a93708effe0fc82cf0d0b94",
    "sie/fmm_periodic.py": "0d97918dd6be22dfc6e4c316ffef85992aac6d581d0bc2eeeda3f2de1ce07458",
    "sie/fmm_periodic_capacity.py": "d13edcbfcdcc717ea0a5812ac9e63fdb061271deeaa60e4d5b37060d8a28c121",
    "sie/fmm_periodic_error.py": "1d2e8cf1fa040c0b1caa5697a7fb84719ff856e7867665a5b4a925965fabf023",
    "sie/fmm_periodic_operator.py": "a630c2de37cfc763ba1a9a9bbd5b457c9bcae59c11748710be99d9927d592715",
    "sie/fmm_periodic_order.py": "a1434e89a9e99118f74b23299acbf054565409e4a62c26e0b0549fc08c5c569a",
    "sie/fmm_periodic_regional.py": "43ae5cc24bd729a8f3745d30720d79662f3616561212803afd83c0c5fb359088",
    "sie/fmm_periodic_system.py": "85cb0443d4cbb26adb80891aad6f622d8860d6bda0c4acdfed1b29e5ecc4c100",
    "sie/fmm_regional.py": "10f00b5bb64c244774c374e886a94c07da08e2463430c3f3ac76a6c897257a48",
    "sie/fmm_rwg_error.py": "106b15fd8f957a763784634b3be4bef812a49bb7f71f68339586791b1bce4b84",
    "sie/fmm_shift.py": "b99d09c94fd8b2760ac747c8c7380263f0ea918cbf23f6f67a95641984a60aab",
    "sie/fmm_storage.py": "6595e6851a7a4354745011a0c1ba647f5fbabc49e42c25b69f9287aa2cee4c74",
    "sie/fmm_system.py": "91c635cb4092786c29715cd4ea6f5d75fd9c94cfceaad4f35eec53eb6885b2fb",
    "sie/fmm_translation.py": "4243d180d2fe42ec1f3de1aaae28a2ac197819a87b4c7532484a16d04652ca17",
    "sie/h2.py": "673189606b7bb984156776afc9d4cee6b02cb0dede19d53ea8d37099655486ad",
    "sie/h2_basis.py": "d28500a776bc981e106acfec101ffb022de8cf69b4947b3fe380a75c441a8c5f",
    "sie/h2_capacity.py": "e76b595a5c8e792dcc48030a27bb5c6e0cf083e5af5f638c1b8414222f210b2d",
    "sie/h2_operator.py": "6bd8155b3ecdc025b7eb15a141d1c80caebe05ad6cb1ebb63c851c7861e767b6",
    "sie/harmonic_balance.py": "ac3c05ec3e289b457b5a6bec2d47a565bc11a49282c0fb79a46e7976a192be2e",
    "sie/harmonic_config.py": "73eb2c91d650074f770f1ccbea15cdd9f028803c93dd57461d7760a52b051e4e",
    "sie/harmonic_convergence.py": "8038d47512031d3eefea22722086d2a6074a51c3db0fc59102fd2f09118d82b0",
    "sie/harmonic_current.py": "7e92ac2a818724eba9e2ab25fb107a2b071bfccb589d7debd8078fad413c75e0",
    "sie/harmonic_current_snapshot.py": "8e333521fa5b942c50048a61992de6413853444cb7c9f25f0cbd5a56616a35a4",
    "sie/harmonic_cutoff_receipt.py": "889e3126ce1a91a942a541de2e727f33280c08547fbe0ee1f5e5b4fbfedb5257",
    "sie/harmonic_dual_bridge.py": "c2505ddbcc84188bd01193329fec46aa7da521fcbef169d82adc95da8e2c67c1",
    "sie/harmonic_dual_current.py": "7ee6f395471c06dd93a32d8fbed39213f69a3ff0d32edbcb34c079e9be5565db",
    "sie/harmonic_fields.py": "9e91fc7ad82049c33602ba1f75a02c39e238985530e693c2848b2d9896bb87f4",
    "sie/harmonic_jacobian_action.py": "4b8c0061c244e1db9a26e1bb8e5ebc72fce8be523cb21efe2d55a48137c393d8",
    "sie/harmonic_material.py": "3bf63c2d88c80e4ec4da74832c5f4368985555bbdf5e8ceb114477d0d1919308",
    "sie/harmonic_mixed.py": "9db118fc02d71095db5dcadecf269f4c320c42526fc74da1105b3f508fa01de3",
    "sie/harmonic_mixed_snapshot.py": "c2cadd9dea25ba50bee81e1a8eba44bb5420dd09f25f27c85282a17d2b1077f4",
    "sie/harmonic_observables.py": "7c5fa4c823a371dfd79e319e8383191e59e59a749acb2048d76b13868b3c2c57",
    "sie/harmonic_preparation.py": "615dad811e77ecd85fbc96aefe0c32843405e17f7190e0e9f2cc087ce1acde1f",
    "sie/harmonic_runtime.py": "ef7b327505db8fe6ee8d122b6639557c565b44d80c7dd5a304722df8d7a257a0",
    "sie/harmonic_snapshot.py": "0bd935a774412455636eb3bf5ecac0cc00d3105d4871e1d720894cfc4975c1aa",
    "sie/harmonic_state.py": "c87147b72b29f7334334f4ee8defaf6ec8fad55e9053cd044a470d9ff291b374",
    "sie/hierarchical_blocks.py": "30cf70df9cc69790ca453bac2900f6d092e0f9dd7561eafce2856f710ed14e63",
    "sie/hierarchical_runtime.py": "504fc0b6d409b4c7fe09140da1bd3c1004b4bba4bcaea52cf89635ea7af30c05",
    "sie/hmatrix.py": "881d6c3bd42281f95db0aaecb88ac5592de008774d916741b1a1ec3b8c13c700",
    "sie/implicit_surface_solve.py": "de9ffad188c8a0b3718f136064ff55472db6583248a7198c3134b935140d2f28",
    "sie/incident_work.py": "f9c2e2c43d08418bf669377efd5d38e4fae30dac585c66e0f5c3b8f4347386bd",
    "sie/interior_volume_hmatrix.py": "4b7bbe967b7df93eab0e742e3eee502084fc24f960b997d0b0e4ca690902de59",
    "sie/interpolated_far.py": "86e76d424325528f5e4eaaf46e34368b44a04e04de845103f636c7bb41b9931f",
    "sie/interpolation.py": "68d34ad5ee3a1937839b84fa5ccf3fd9352f984bd8ca8a173b42a21d0ae50df1",
    "sie/interpolation_error.py": "f800139ce352dfbcd867116d36f5bccfd7514c8d120b3090a3b383c0efc220ab",
    "sie/interpolation_kernel.py": "7293d9ac75a9ab0c94e57e64b6c762360e78866fb69e582d01f035bcb1840caa",
    "sie/interpolation_order.py": "1db1e1576cbf926fb2025a61a91a10eabfed26e86dc539899965cfeba23bfcec",
    "sie/kernels.py": "9bc5d67e6763eb17d8b804a23dbbaa2a3aebafbf054cd70bdfdff6aaeb184f6c",
    "sie/lifting.py": "cb5b684595a5801c9090b4ad5ed624a34a8a19bae44779f39e50611d927d290a",
    "sie/lifting_fields.py": "6fc073371cf8d8370924124515ae8bc412f41deaf25b180509590e37305c4629",
    "sie/linalg.py": "a75d3c5533e8824222f72d8895d14e543eca065024e563656ba87f42b9d302e0",
    "sie/linear_runtime.py": "e08a325634a5a67ea70a4acf104ab3092cc68aec73ee1d326184e4798ab160ce",
    "sie/linear_volume_state.py": "1fce1e540e4a973180a0ecaadc6c924cdb28a51caedab11b5de8925deb07707f",
    "sie/lowrank.py": "4cbac4fb3b4bebb9306eca48469c54054ac2e372695045376e668f4b6fc48a14",
    "sie/magnetostatic.py": "a00dc77815daa3ac7c7e0103e1baae79cb243437ee799b15c1d935e77ddc759b",
    "sie/matched_spd_green.py": "783fafef6803e8c88184d9dded89fe60babab19427ad962a247d2bfa262ce0bd",
    "sie/mesh.py": "722b928a72220851e1f04c318b959c4a89c473a8d73c85233045955b09ad17f7",
    "sie/migration.py": "88b3a7dacd2697a26b5aad3bb3ff9c892d0e8a8d0048615b51432f0b0d7fd28e",
    "sie/modes.py": "b95b2797c7b13824379ebd6f9ee64cb13f2585a0358480530bd1950626280f2e",
    "sie/near_preconditioner.py": "1d04feed461ef6956bdd32707a1914bb9a01407a69f2717e351529aa3ed7f6c3",
    "sie/nearfield.py": "c43ac744f4c3b944ffdf157d539d0b85c96f6650d27caf96a82cbc5773476ee0",
    "sie/nested_h2_snapshot.py": "656ae8e714f46ec7be9826ea53e0b4f4f638f9043b562b24abd42ef65f5ffcfb",
    "sie/nested_moments.py": "56560a17b911392c27bd021785337822b0d4d50ebb101653c6c369b761d6ceb7",
    "sie/nonlinear.py": "9224335c0313489d72aa46b225ae5a5ed03ff28c0d651890f5bd8e3ef91321f3",
    "sie/observables.py": "aae8608a4a30959945e738fbc69b3fbc5805cf30df232bfb96c542b3915e6591",
    "sie/open_surface.py": "e1ee9805050040f0ca11cbc1c73859671059b4f27144b4b8046b7958840cd5db",
    "sie/operator.py": "6ea489918eac160f64f6558e31af26fabdf7ed0e0d1efdb1f9b06ca934ac152f",
    "sie/operator_error.py": "e132bcb9c39d29547421913df6ea6d42b2f1526a160959c6348dfc7da51dce60",
    "sie/operator_protocol.py": "ce8a9ec8441857151021321cb8537b754d5de6d511fa46e8897e91722c886037",
    "sie/ordered_reduction.py": "0b74eb7f5b62ea98ffd35afa3f01bd811e8240e378c0d0d33727ec7bf16896ff",
    "sie/packed_near.py": "6e2b72c151f301bb3bb2e12e6037d53d1325dbaca70e7456f33f55ca73364668",
    "sie/periodic.py": "18e449d2b69ec409e1c40a06bae3551e63a58c4ea087d0169bd99186eb840629",
    "sie/periodic_config.py": "a2eca9d267736c02fb992ecad16006c5e26b2429830783a8f8103fcd07a7a094",
    "sie/periodic_cylindrical.py": "cc7bc7da1d07e4e0a7aa690f7f310421797ce04493f4f1b578843911db725404",
    "sie/periodic_diffraction.py": "22bdb43184e66f99355ddbd58612916c5de19c07b558efb0b0b91090dd2f993a",
    "sie/periodic_dyadic.py": "667d4eea4f69d743b5aa96ac32fa4954aa0bb071fc876c3e822f02590d087aa9",
    "sie/periodic_dyadic_fields.py": "dc56c24efaad62f21038fc74a4928784c7a8971fbf143c6c71b7f8514d738442",
    "sie/periodic_dyadic_nearfield.py": "74acb75f8b0960eb20733891c07dd8511032586baf37dc8668da7bba4534f88f",
    "sie/periodic_dyadic_operator.py": "4930918c918b793faa608a174d676f4c47f223591bc0c8800a328fca2ef844f1",
    "sie/periodic_fields.py": "dbac30c007be64e706b409177d1cb0bd6533ce825c8615b5bd2369311c889851",
    "sie/periodic_hierarchical_runtime.py": "a2f1c7378415851ef69b44b9ccf584fb61a943384436d6d3f21df0d368031895",
    "sie/periodic_limits.py": "a3b710b6d4c3b99182aae93fd5197ea946f13bf88db3ae6e6274d800f645049e",
    "sie/periodic_nearfield.py": "00bb6825fe50eb2d1c5f421226ec78a550741b2c4222194efa734d2f2fa9da24",
    "sie/periodic_operator.py": "614ed766187f5cb3d49caefa559b18b5af960734d04bc711e42e514253d54133",
    "sie/periodic_runtime.py": "0f972797d813313fa821b4fdf45249df17be7cf78958b9318df0ef41f3629480",
    "sie/periodic_spectral.py": "63e3de5e9de867099b9544675f0b2a33eb2d0b44e4dcff0f7bd57abd8a4651fc",
    "sie/pfft_config.py": "3715a018834a0fee56746c50859354b6943babbedac2d73d78fecf41f96ce031",
    "sie/pfft_coupling.py": "0da99473879be3db3b8a56a7de8817b2d7d4696aa1ede18ed002d1f3fdeaf863",
    "sie/pfft_efie.py": "a4220ec07b83d0a6d6752cbd1e99f412bef81db43649aefda2cf880bef6bcc3b",
    "sie/pfft_grid.py": "f30c70e028fde46aff97553a69ba95deca3a3d5cdc9b992934e2f758e5dc7b5a",
    "sie/pfft_lifting.py": "02a922594e8988a3833e5a13ec723c0b2000dd49e75f32b97bfcebfa69fa6508",
    "sie/pfft_operator.py": "489409574319b8f622c6f862fe73f7e2a3a0c345b443878934d8308f48e0bee7",
    "sie/pfft_system.py": "4259e82a9d0bc5d36b5f08f7ae3d0569972f1ed3130bdd41218711f2f75cdd7a",
    "sie/physical_surface_snapshot.py": "afb32c212de03d34a95e02800be0fef59f1cce49b0fae89eb7fedd50b4f97424",
    "sie/physics.py": "0f268909e84c9d1622283d6009c3cc587d4a5635217285112069d1a36dd84c9e",
    "sie/preconditioning.py": "678d099c3b2126df4e0ee4e98f2120b5bd901c211bd603775fbf16418ef2e33a",
    "sie/primitive_mesh.py": "2d27ce26afad7609e3decc05776e459099e6625a11fd7dcb9c323ed65a1a8b20",
    "sie/quadrature.py": "85129e2710f724e524f9972e80caba3ac16ce6f20cdccf0850744a12729c86be",
    "sie/riesz.py": "82139e24b3d94b0395fc941e4359c33529d868626c7da51501f0580e582b3e1a",
    "sie/runtime.py": "755299f12b1d313d6005e7f2492f9d69a44dc41ddc63a712468eba5c1e959141",
    "sie/rwg_interpolation_error.py": "a890d321658c75bbfcccc9d97c65336ccea4ada8719e4af298883a075ccce47e",
    "sie/rwg_moments.py": "8831f55fc2027e136f8514ab95216b16108aa580f1c553bfee933c95586a28fa",
    "sie/rwg_refinement.py": "7285ea632a3f36042d5d18ae627625cbaed81a347786b293bb167311eef90f0f",
    "sie/scalar_moment_action.py": "9dac838b9775fb59835313e276b0dc4be1eea965b7a7dcc58c2844b8a9fbd341",
    "sie/scalar_near_channels.py": "f59729b1a7b4afcd9a730f8174e0bea2290c93fdc05cabd4c93d6b7833c05ca7",
    "sie/single_layer.py": "d14c00697b4a207853834ef2875f56ef3dc7071d825c36142dcb6f53f680c28f",
    "sie/spatial.py": "2a0f38c4fc93cf8cb8d04d9570fca528fc85dce39be921b5e62367a8b9643287",
    "sie/spectral_linalg.py": "5f15c2db45a844bf44a61d1bed351bd685b0efaf32343cca1fa405851d0c056d",
    "sie/static_coupling.py": "c753b1818bce9214a65abe8948fd4a0570cad53223516a7a5232a7913e072d58",
    "sie/static_current.py": "2e6432a62d4ac49e89bf43b410c78d102b2dab5539ea2ea1a63ca436c55eb50d",
    "sie/static_current_projector.py": "d1da3655ce8459f25e51c4607709e65eea5187133cf29dda0f77ca72766060e6",
    "sie/static_floating.py": "6887ae045bfa6f8317d052671642232d71d3909fbb0f3b06f1a8dba990f0677b",
    "sie/static_magnetic.py": "8a5e71f9a1edc1827ed14f7288af2654a2234fee2afab6648de49123c1522371",
    "sie/static_magnetic_snapshot.py": "3183300e25a95db8b62d7ea638501718ffeee868882e0771446cece69aabf524",
    "sie/static_magnetic_volume.py": "8f07dad93d1d957d26a4f28061bd2314b510bd619c1e2561f87b840ec34893da",
    "sie/static_mixed.py": "dbaa422d851e34faef7bcdeb6695bb1b01eee2a01dd3546d754c868b107082e2",
    "sie/static_projector_snapshot.py": "6576501e259dbcdf21cbf958c5344d2e6deb3f757020158d58da3940ecd78a8c",
    "sie/static_tensor_floating.py": "75d759f2ddcd142dfde0d9ec8f17cd272be704b0fe4f501f4a68ee2614fe554f",
    "sie/static_tensor_green.py": "72c65973b1a6694e86d78cf0c154a0e865d21345d9d6b32b43be971a50b1806e",
    "sie/static_tensor_interface.py": "b9ab722c5dab4741d73351d41d286b3f4253f77c9d66516625809fda2e28a5e3",
    "sie/static_tensor_material.py": "87730d27a3b38b12398f6141c83ee06a287cb620888d8ab1195d679af949cca4",
    "sie/static_tensor_polarization.py": "f8be1696d8ef20ceb672d5a17f0f741e20cdacd67467730b6449f78e9b6e4852",
    "sie/static_tensor_preparation.py": "bbbb095087463671c76ed511e14a4bd53e37ca78be8192cfc91de0dd4c03285a",
    "sie/static_tensor_snapshot.py": "9a758f447b9087c7988b8848f71f31537c7f57dfc3fac593bac66aee56886f2e",
    "sie/static_tensor_volume.py": "bf08d3af8df225e8f1220dc752d907942eee121cb4c4fbb275ad50c183873fe5",
    "sie/static_volume_curl.py": "a61464ac8a784418951d1d27857bb142113f407b2d012e9ca9e05f7a3fd1b445",
    "sie/streamed_dual_state_snapshot.py": "4514a2412979437019e12c1627739a8d62bcebcbe6a5c29ac827e21570b98e2d",
    "sie/surface_h_snapshot.py": "198ddff5ccd912530514917dd4ff77b30e670489e6ea3a01b86ee3b2b7b4b111",
    "sie/surface_volume.py": "0846eae073968bc252d29cfa8e29f595efd68aa8dc33a3bf7b42dbd19cf66ce5",
    "sie/surface_volume_hmatrix.py": "fcb771316fdfe96f8c6a7dbad80d43d989acb160b5719ba6134f057142e17aec",
    "sie/tensor_carrier.py": "24d38aa16f54339538f33d79a670baf1d6157a8d5e21402cba7a22f4db16c4ad",
    "sie/tensor_harmonic_bundle.py": "7e62f3df03712d8cc49cb3a69a0b3316eb1f0a26493fc0ef1730f02d8b10c6f3",
    "sie/tensor_harmonic_preparation.py": "9b6d198778bf3c11660afe3daf54888d987d96f06ece4403f4a9beac96ab1bb7",
    "sie/tensor_harmonic_runtime.py": "1bf4138c613659560f4054075c48218de386a005f041fc1eed6784c3097fad9d",
    "sie/tensor_harmonic_snapshot.py": "4461de07880d1bfe744567060b58ca5cdf7fdbb512cf604feb218ec4692e8dcb",
    "sie/tensor_medium.py": "f5a57abf091f60bea6b7de9b920ee95d547fad353a155276adf813c424706d70",
    "sie/tensor_nearfield.py": "f267a067bf0ab2fad66be0b32e5fb2dfaab9ee027232a595f1a564a1017aab66",
    "sie/tensor_volume.py": "97cac21d9fad4e0b2477006243ae9fb091d4f7c9c21d14e2da27c07284062c9c",
    "sie/topology.py": "8c55779aaf89a90863ae7bd52540e3c77236f48a66888a8f50f5a2e642214414",
    "sie/vie_operator.py": "707c23358af7e3612cce3d330f17a0940770ea29a8990f98fd355e3d8f77f584",
    "sie/volume_basis.py": "1fd511c542a05c56c09e7ecbc9db9a5df178e5c49b3a3fe883299449547c9d3c",
    "sie/volume_h2.py": "0620aa567312fc2025c8a5f680db1f029293ac7bea3778638f9fb7495606e44b",
    "sie/volume_h2_snapshot.py": "b3213f0990a5dfa92f0111340da3e3fe1e3f644c5063aa83921f6dd010798c8a",
    "sie/volume_h_snapshot.py": "6a1f78acb21c4000c1cb2e1fc280d3ca4b80b40bd12d41e4b61ddc6c43db39c5",
    "sie/volume_hmatrix.py": "b1d9c29a1b533c9dba65f5654ac21583a3d53107e7cb25a4f4e2625b2c9b2105",
    "sie/volume_mesh.py": "60f204d08961d0ad6bc39ccec58a4e20de58149269cf976290b51b258ededa2e",
    "sie/volume_response_matrix.py": "e7b546f733c393d5d9557331ecd0baaa8ad1a0afca15efed4b43a00d54c1c50d",
    "sie/volume_source_refinement.py": "fd38f15cabf31b7a332fcc66f4144a601c25293578e74c10632ff3c5e3ad96e1",
    "simulation.py": "0a3211567090cd795863cc88d977655a7372b57e664511d44080f3a4e68f8dc1",
    "sources.py": "9f03c0f1825d7661dedb148f8da7ef89e200d48e4f11f235e80104b3312fcaaa",
    "terminal_waveport.py": "859965943da81e804098145f9bfc4c52069c8f65c7c589eda17cb246b0bfbae9",
    "waveforms.py": "66f4a3bd7b5a601f9d000a93599f69ee1e056162fa6f31ab44a938c1181fde5b",
    "waveport_sweep.py": "f7864d0d46b06291dbb19511f8c94f2049c1a1c8528e1aff03e8f6adffc822e5"
}

def load_binding_helper(driver,expected):
 data=Path(driver).read_bytes()
 if expected!=BINDING_HELPER_SHA256 or sha(data)!=BINDING_HELPER_SHA256:raise ValueError('Wrong reviewed equivalence driver')
 spec=importlib.util.spec_from_file_location('reviewed_equivalence_driver',driver)
 module=importlib.util.module_from_spec(spec)
 exec(compile(data,str(driver),'exec'),module.__dict__)
 return module

def sha(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def canonical(value):return sha(json.dumps(value,sort_keys=True,separators=(',',':')).encode())

class BenchmarkSources(importlib.abc.MetaPathFinder):
 """Bind only this candidate's benchmark namespace and Python modules."""
 def __init__(self,root,bound):
  if any(n=='benchmark' or n.startswith('benchmark.') for n in sys.modules):raise ValueError('benchmark already imported before archive binding')
  self.root=(Path(root)/'benchmark').resolve();self.bound=bound;self.namespaces={};self.loaded={};self.package_paths={}
 def find_spec(self,fullname,path=None,target=None):
  if fullname!='benchmark' and not fullname.startswith('benchmark.'):return None
  spec=importlib.machinery.PathFinder.find_spec(fullname,path)
  if spec is None:raise ValueError('Unarchived benchmark module resolution')
  owner=self
  if spec.origin is None:
   locations=tuple(str(Path(p).resolve()) for p in (spec.submodule_search_locations or []))
   if fullname!='benchmark' or locations!=(str(self.root),):raise ValueError('Foreign benchmark namespace path')
   class NamespaceLoader(importlib.abc.Loader):
    def create_module(self,spec):return None
    def exec_module(self,module):owner.namespaces[fullname]=(module,locations)
   spec.loader=NamespaceLoader();return spec
  origin=str(Path(spec.origin).resolve())
  if not Path(origin).is_relative_to(self.root) or origin not in self.bound.files or not origin.endswith('.py'):raise ValueError('Unarchived benchmark import origin')
  class Loader(importlib.abc.Loader):
   def create_module(self,spec):return None
   def exec_module(self,module):
    row=owner.bound.files[origin];data=Path(row['archive']).read_bytes()
    if sha(data)!=row['sha256'] or sha(Path(origin).read_bytes())!=row['sha256']:raise ValueError('Benchmark source changed before first execution')
    module.__file__=origin;exec(compile(data,origin,'exec'),module.__dict__)
    owner.loaded[fullname]=(module,origin);owner.bound.modules[fullname]=(module,origin)
    if hasattr(module,'__path__'):owner.package_paths[fullname]=(module,(str(Path(origin).parent),))
  spec.loader=Loader();return spec
 def verify(self):
  actual={n for n in sys.modules if n=='benchmark' or n.startswith('benchmark.')}
  if actual!=set(self.loaded)|set(self.namespaces):raise ValueError('Unbound benchmark module appeared')
  for origin,row in self.bound.files.items():
   if Path(origin).is_relative_to(self.root) and (sha(Path(origin).read_bytes())!=row['sha256'] or sha(Path(row['archive']).read_bytes())!=row['sha256']):raise ValueError('Benchmark source/archive changed')
  for name,(obj,origin) in self.loaded.items():
   if sys.modules.get(name) is not obj or self.bound.modules.get(name)!=(obj,origin):raise ValueError('Benchmark module object replaced')
  for name,(obj,locations) in (self.namespaces|self.package_paths).items():
   if sys.modules.get(name) is not obj or tuple(str(Path(p).resolve()) for p in obj.__path__)!=locations:raise ValueError('Benchmark namespace replaced')
  return {'loaded_modules':{n:{'origin':o,'sha256':self.bound.files[o]['sha256'],'module_object_id':id(obj)} for n,(obj,o) in self.loaded.items()},'namespaces':{n:{'paths':list(p),'module_object_id':id(obj)} for n,(obj,p) in self.namespaces.items()}}


def grade(nodes,phases,per_test,exitstatus,source_ok,devices,required_cuda_nodes=()):
 errors=[]
 if exitstatus!=0:errors.append('pytest nonzero exit')
 if not source_ok:errors.append('source or initial module binding changed')
 if len(nodes)!=len(set(nodes)) or not nodes:errors.append('empty/duplicate collection')
 if [x['nodeid'] for x in per_test]!=nodes:errors.append('executed test protocols differ from collection')
 expected=[(n,p) for n in nodes for p in ('setup','call','teardown')]
 if [(x['nodeid'],x['when']) for x in phases]!=expected:errors.append('missing/duplicate/out-of-order phases')
 if any(type(x.get('duration')) not in (int,float) or not math.isfinite(x['duration']) or x['duration']<0 for x in phases):errors.append('malformed phase duration')
 if any(x['outcome']!='passed' or x.get('wasxfail') is not None for x in phases):errors.append('failed/skipped/xfail/xpass phase remains')
 observed={d:0 for d in devices}
 for t in per_test:
  if t.get('measurement_errors'):errors.append('CUDA measurement failed')
  if set(t.get('cuda',{}))!=set(devices):errors.append('missing requested device observation')
  if type(t.get('seconds')) not in (float,int) or not math.isfinite(t['seconds']) or t['seconds']<=0:errors.append('malformed test duration')
  for d,row in t.get('cuda',{}).items():
   if d not in observed or any(type(row.get(k)) is not int or row[k]<0 for k in ('allocation_events','allocated_before','allocated_after','peak_allocated_bytes','peak_reserved_bytes')):errors.append('malformed actual CUDA counters');continue
   if row['peak_allocated_bytes']<max(row['allocated_before'],row['allocated_after']):errors.append('inconsistent actual CUDA peak')
   observed[d]+=row['allocation_events']
 if any(n not in nodes for n in required_cuda_nodes):errors.append('Required CUDA node absent from collection')
 by_node={r['nodeid']:r for r in per_test}
 for node in required_cuda_nodes:
  row=by_node.get(node,{})
  if sum(v.get('allocation_events',0) for v in row.get('cuda',{}).values() if type(v.get('allocation_events')) is int)<=0:errors.append('Required direct CUDA node has no actual allocation: '+node)
 return {'passed':not errors,'errors':errors,'actual_test_allocation_events_by_device':observed}

def verify_junit(path,nodes):
 from _pytest.junitxml import mangle_test_address
 cases=ET.parse(path).findall('.//testcase')
 expected=[]
 for node in nodes:
  parts=mangle_test_address(node);expected.append(('.'.join(parts[:-1]),parts[-1]))
 if [(x.get('classname'),x.get('name')) for x in cases]!=expected:raise ValueError('JUnit exact ordered testcase identities differ')
 if any(x.find(k) is not None for x in cases for k in ('failure','error','skipped')):raise ValueError('JUnit contains a nonpassing testcase')
 return {'exact_ordered_testcases':len(cases),'sha256':sha(Path(path).read_bytes())}


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def commit_inventory(root, revision):
    raw = git(root, 'archive', revision, 'witwin/maxwell')
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        return {m.name.removeprefix('witwin/maxwell/'): {
            'sha256': sha(archive.extractfile(m).read()), 'bytes': m.size}
            for m in archive.getmembers() if m.isfile()}


def file_record(path):
    return {'sha256': sha(Path(path).read_bytes()), 'bytes': Path(path).stat().st_size}


def verify_source(root):
    old = commit_inventory(root, BASE_COMMIT)
    frozen = commit_inventory(root, SOURCE_COMMIT)
    changed = {n: v['sha256'] for n, v in frozen.items() if old.get(n) != v}
    if changed != REVIEWED_DELTAS or set(old) - set(frozen):
        raise ValueError('Package delta differs from the 369 reviewed source paths')
    actual = {str(p.relative_to(root / 'witwin/maxwell')): file_record(p)
              for p in sorted((root / 'witwin/maxwell').rglob('*'))
              if p.is_file() and p.suffix in SOURCE_SUFFIXES and '__pycache__' not in p.parts}
    if actual != frozen:
        raise ValueError('Live provider package differs from frozen source, including added paths')
    tree = git(root, 'rev-parse', SOURCE_COMMIT + ':witwin/maxwell/sie').decode().strip()
    if tree != SIE_TREE:
        raise ValueError('Frozen SIE tree differs from the reviewed numerical tree')
    subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', SOURCE_COMMIT, 'HEAD'], check=True)
    return dict(baseline_commit=BASE_COMMIT, source_commit=SOURCE_COMMIT, sie_tree=tree,
                baseline=old, candidate=actual, reviewed_scientific_deltas=changed)


def inventory(root, core, policy, policy_path, costs_path):
    paths = set()
    for directory in (root / 'witwin/maxwell', core, root / 'tests/sie',
                      root / 'benchmark', root / 'scripts'):
        paths.update(p.resolve() for p in directory.rglob('*')
                     if p.is_file() and p.suffix in SOURCE_SUFFIXES and '__pycache__' not in p.parts)
    for name in ('tests/__init__.py', 'tests/api/public/test_public_api.py', 'tests/conftest.py',
                 'tests/api/conftest.py', 'tests/api/public/conftest.py', 'pyproject.toml',
                 'pytest.ini', 'setup.cfg', 'tox.ini'):
        p = root / name
        if p.is_file():
            paths.add(p.resolve())
    paths.update(p.resolve() for p in (root / 'docs/reference').rglob('*')
                 if p.is_file() and p.suffix in {'.json', '.md', '.txt'})
    frozen_root = core.parents[1]
    freeze_path = frozen_root.parent / 'core-freeze.json'
    paths.add(freeze_path)
    paths.update(frozen_root / p for p in read(freeze_path)['files'])
    paths.update(Path(p).resolve() for p in policy['files'] if Path(p).is_file())
    paths.update((Path(__file__).resolve(), policy_path.resolve(), costs_path.resolve()))
    paths.add(Path(read(costs_path)['source']['path']))
    return {str(p): file_record(p) for p in sorted(paths)}


def logical_inventory(files, root, core):
    result = {}
    for name, row in files.items():
        p = Path(name)
        if p.is_relative_to(core):
            key = 'core/' + str(p.relative_to(core))
        elif p.is_relative_to(root):
            key = 'provider/' + str(p.relative_to(root))
        else:
            key = 'external/' + name
        if key in result:
            raise ValueError('Duplicate logical source identity')
        result[key] = row
    return result


def bind_frozen_core(frozen_root, root):
    frozen_root = frozen_root.resolve()
    manifest = read(frozen_root.parent / 'core-freeze.json')
    actual = {str(p.relative_to(frozen_root)): file_record(p)
              for p in sorted(frozen_root.rglob('*')) if p.is_file()}
    if actual != manifest['files'] or canonical(actual) != manifest['frozen_inventory_sha256']:
        raise ValueError('The frozen core copy differs from its byte-identical snapshot')
    os.environ['PYTHONPATH'] = os.pathsep.join((str(frozen_root), str(root)))
    sys.path[:0] = [str(frozen_root), str(root)]
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    core = Path(importlib.util.find_spec('witwin.core').origin).resolve().parent
    distribution = importlib.metadata.distribution('witwin')
    if (core != frozen_root / 'witwin/core' or not Path(distribution._path).is_relative_to(frozen_root)
            or distribution.version != manifest['version']):
        raise ValueError('Core import/distribution escaped the frozen execution copy')
    return core, manifest


def load_costs(path):
    manifest = read(path)
    raw = Path(manifest['source']['path'].replace('<evidence-root>', os.environ.get('WITWIN_EVIDENCE_ROOT', 'evidence'))).read_bytes()
    if sha(raw) != manifest['source']['sha256']:
        raise ValueError('Historical timing ledger changed')
    manifest['nodes'] = {n: dict(seconds=sum(p['duration_s'] for p in r.get('reports', [])),
                                outcome=r['outcome'])
                         for n, r in json.loads(raw).items()
                         if n.startswith(('tests/sie/', 'tests/api/public/test_public_api.py'))}
    return manifest


def artifact_audit(root, policy):
    problems = []
    for name, expected in policy['files'].items():
        p = Path(name)
        if not p.is_file():
            problems.append(dict(path=name, reason='missing_original_input'))
        elif sha(p.read_bytes()) != expected:
            problems.append(dict(path=name, reason='changed_original_input'))
    for name, expected in policy['helper_dependency_source_pins'].items():
        if sha((root / name).read_bytes()) != expected:
            problems.append(dict(path=name, reason='changed_helper_requires_closure_review'))
    skip_sites = []
    for path in (root / name for name in SCOPE):
        tree = ast.parse(path.read_text())
        keys = sorted({n.value for n in ast.walk(tree)
                       if isinstance(n, ast.Constant) and isinstance(n.value, str)
                       and n.value.startswith('GSP_')})
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == 'pytest'
                    and node.func.attr == 'skip'):
                skip_sites.append(dict(module=str(path.relative_to(root)), line=node.lineno,
                    expression=ast.unparse(node), module_environment_keys=keys,
                    supplied_keys=sorted(policy['modules'].get(str(path.relative_to(root)), {}).get('environment', {}))))
    return dict(problems=problems, conditional_skip_sites=skip_sites,
                original_pins=len(policy['files']), module_policies=len(policy['modules']),
                unqualified_artifact_nodes=ARTIFACT_UNAVAILABLE_NODES,
                ready_for_execution=not problems,
                scope='Host availability and static conditional skips; test bodies were not executed')


@contextlib.contextmanager
def module_environment(policy, module):
    previous = {k: v for k, v in os.environ.items() if k.startswith('GSP_')}
    for key in previous:
        del os.environ[key]
    os.environ.update(policy['modules'].get(module, {}).get('environment', {}))
    try:
        yield
    finally:
        for key in list(os.environ):
            if key.startswith('GSP_'):
                del os.environ[key]
        os.environ.update(previous)


def partition(nodes):
    if any(n.split('::')[0] not in SCOPE for n in nodes):
        raise ValueError('Collected nodes escaped the reviewed dense-core/default modules')
    expected = MULTI_DEVICE_NODES[0]
    if nodes.count(expected) != 1:
        raise ValueError('The selected module lost its reviewed two-device cache control')
    if any(nodes.count(node) != 1 for node in ARTIFACT_UNAVAILABLE_NODES):
        raise ValueError('The reviewed artifact-unavailable node identity changed')
    excluded = (*MULTI_DEVICE_NODES, *ARTIFACT_UNAVAILABLE_NODES)
    return [n for n in nodes if n not in excluded], list(MULTI_DEVICE_NODES), []


def make_shards(nodes, costs):
    shards = []
    for node in nodes:
        module = node.split('::')[0]
        observed = costs['nodes'].get(node)
        seconds = max(1., observed['seconds'] * 1.5) if observed else 60.
        if seconds + 30 > 885:
            raise ValueError('An indivisible node exceeds the 885-second lane reservation: ' + node)
        if (not shards or shards[-1]['module'] != module or len(shards[-1]['nodeids']) >= 16
                or shards[-1]['estimated_seconds'] + seconds > 720):
            shards.append(dict(id=len(shards), module=module, nodeids=[], estimated_seconds=30.,
                               locks=1,
                               memory_reservation_bytes=20_000_000_000,
                               torch_allocator_limit_bytes=18_000_000_000,
                               historical_nodes=0, unmeasured_nodes=0))
        row = shards[-1]
        row['nodeids'].append(node)
        row['estimated_seconds'] += seconds
        row['historical_nodes' if observed else 'unmeasured_nodes'] += 1
    for row in shards:
        row['timeout_seconds'] = 885
    return shards


def check_catalog(catalog, files, root, core):
    if (catalog['protocol'] != PROTOCOL or catalog['scope'] != SCOPE
            or catalog['source_inventory_sha256'] != canonical(logical_inventory(files, root, core))
            or catalog['harness_sha256'] != sha(Path(__file__).read_bytes())):
        raise ValueError('Catalog source/protocol/harness binding changed')
    nodes, multi, hash_only = partition(catalog['full_nodeids'])
    if (nodes != catalog['ordered_nodeids'] or multi != catalog['unqualified_multi_device_nodes']
            or catalog['unqualified_artifact_nodes'] != ARTIFACT_UNAVAILABLE_NODES
            or hash_only != catalog['hash_only_backend_nodes']
            or [n for shard in catalog['shards'] for n in shard['nodeids']] != nodes
            or len(nodes) != len(set(nodes))):
        raise ValueError('Catalog partition or exact disjoint shard coverage changed')


@contextlib.contextmanager
def gpu_locks(count):
    import fcntl
    paths = [Path(os.environ['WITWIN_GPU_LOCK_DIR']) / name
             for name in ('cuda1.lock', 'cuda1c.lock')]
    if count != 1:
        raise ValueError('Only one A/C lane is authorized; larger jobs must stop')
    with contextlib.ExitStack() as stack:
        for p in paths:
            stream = p.open('r')
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            stack.enter_context(stream)
            yield [str(p)]
            return
        raise BlockingIOError('Neither authorized GPU-1 lane A nor C is free')


@contextlib.contextmanager
def shard_deadline(seconds):
    def expired(signum, frame):
        raise TimeoutError('The reviewed shard time limit expired')
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextlib.contextmanager
def process_memory_limit(limit_bytes):
    owner = os.getpid()
    stop = Event()
    state = dict(peak_process_bytes=0, samples=0, device=1, pid=owner)

    def sample():
        while not stop.is_set():
            raw = subprocess.check_output(['nvidia-smi', '--id=1',
                '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader,nounits'],
                text=True, timeout=2)
            for line in raw.splitlines():
                pid, mib = line.split(',')
                if int(pid.strip()) == owner:
                    value = int(mib.strip()) * 1024**2
                    state['peak_process_bytes'] = max(state['peak_process_bytes'], value)
                    if value > limit_bytes:
                        raise MemoryError('GPU-1 process usage exceeded its 20 GB lane limit')
            state['samples'] += 1
            stop.wait(.2)

    def failed(signum, frame):
        raise RuntimeError('GPU-1 process-memory monitor failed: ' + str(future.exception()))

    previous = signal.signal(signal.SIGUSR1, failed)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(sample)
        future.add_done_callback(lambda f: os.kill(owner, signal.SIGUSR1) if f.exception() else None)
        try:
            yield state
        finally:
            stop.set()
            future.result()
            signal.signal(signal.SIGUSR1, previous)


def aggregate(args):
    catalog = read(args.catalog)
    if sha(args.catalog.read_bytes()) != args.reviewed_catalog_sha256:
        raise ValueError('Aggregate requires the exact user-reviewed catalog')
    runs = [read(p / 'execution.json') for p in args.shard_results]
    if [r['shard_id'] for r in runs] != [s['id'] for s in catalog['shards']]:
        raise ValueError('Aggregate lacks exact ordered shards or includes duplicates')
    for run, directory, shard in zip(runs, args.shard_results, catalog['shards']):
        if (run['catalog_sha256'] != args.reviewed_catalog_sha256 or run['status'] != 'passed'
                or run['source_inventory_sha256'] != catalog['source_inventory_sha256']
                or run['ordered_nodeids'] != shard['nodeids']):
            raise ValueError('Unsuccessful or different-epoch shard cannot enter aggregation')
        for name, expected in run['artifacts'].items():
            if file_record(directory / name) != expected:
                raise ValueError('Shard artifact changed before aggregation')
        if verify_junit(directory / 'junit.xml', shard['nodeids']) != run['junit_verification']:
            raise ValueError('Shard JUnit binding changed')
    phases = [p for r in runs for p in r['phases']]
    tests = [p for r in runs for p in r['per_test']]
    verdict = grade(catalog['ordered_nodeids'], phases, tests, 0, True, ['0'], catalog['required_cuda_nodeids'])
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(protocol=PROTOCOL, mode='aggregate', status='passed' if verdict['passed'] else 'failed',
        engineering_acceptance_passed=verdict['passed'], verdict=verdict, ordered_nodeids=catalog['ordered_nodeids'],
        phases=phases, per_test=tests, catalog_sha256=args.reviewed_catalog_sha256,
        source_inventory_sha256=catalog['source_inventory_sha256'],
        unqualified_multi_device_nodes=catalog['unqualified_multi_device_nodes'],
        unqualified_artifact_nodes=catalog['unqualified_artifact_nodes'],
        hash_only_backend_nodes=catalog['hash_only_backend_nodes'],
        shards=[dict(path=str(p), sha256=sha((p / 'execution.json').read_bytes())) for p in args.shard_results],
        gpu_seconds=sum(r['elapsed_seconds'] for r in runs), H2_G0='not_assessed')
    write(args.output / 'execution.json', result)
    return 0 if verdict['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('collect', 'run', 'aggregate'), required=True)
    parser.add_argument('--candidate', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--reviewed-catalog-sha256')
    parser.add_argument('--shard', type=int)
    parser.add_argument('--shard-results', type=Path, nargs='*')
    parser.add_argument('--core-root', type=Path)
    args = parser.parse_args()
    if Path(sys.executable).resolve() != Path(PYTHON).resolve():
        raise ValueError('Use the pinned maxwell interpreter')
    if args.mode == 'aggregate':
        return aggregate(args)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != ('' if args.mode == 'collect' else '1'):
        raise ValueError('Collect hides CUDA; run exposes only physical GPU 1')
    root = args.candidate.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(mode=args.mode, protocol=PROTOCOL, status='failed', engineering_acceptance_passed=False,
                  H2_G0='not_assessed', pid=os.getpid(), argv=sys.argv, source_origin=str(root))
    evidence = None
    try:
        pair = verify_source(root)
        report['provider_pair'] = pair
        report['execution_head'] = git(root, 'rev-parse', 'HEAD').decode().strip()
        core, core_manifest = bind_frozen_core(args.core_root, root)
        os.chdir(root)
        driver = root / 'benchmark/sie_single_buffer_equivalence_v3.py'
        helper = load_binding_helper(driver, BINDING_HELPER_SHA256)
        policy_path = root / 'benchmark/sie_combined_full_acceptance_v3_inputs.json'
        costs_path = root / 'benchmark/sie_combined_full_acceptance_v3_costs.json'
        policy, costs = read(policy_path), load_costs(costs_path)
        audit = artifact_audit(root, policy)
        before = inventory(root, core, policy, policy_path, costs_path)
        logical = logical_inventory(before, root, core)
        write(out / 'source-before.json', before)
        write(out / 'artifact-audit.json', audit)
        bound = helper.BoundSources([Path(p) for p in before], out / 'source')
        sys.meta_path.insert(0, bound)
        benchmark = BenchmarkSources(root, bound)
        sys.meta_path.insert(0, benchmark)
        write(out / 'provider-pair.json', pair)
        catalog = read(args.catalog) if args.mode == 'run' else None
        if args.mode == 'run':
            if sha(args.catalog.read_bytes()) != args.reviewed_catalog_sha256:
                raise ValueError('Run requires the exact user-reviewed catalog SHA256')
            check_catalog(catalog, before, root, core)
            if not audit['ready_for_execution']:
                raise ValueError('Original artifact audit is blocked; see artifact-audit.json')
            if catalog['estimated_gpu_seconds'] > catalog['user_gpu_budget_seconds']:
                raise ValueError('The frozen campaign cost estimate exceeds the approved eight-hour budget')
            shard = catalog['shards'][args.shard]
            report.update(shard_id=args.shard, catalog_sha256=args.reviewed_catalog_sha256)
        else:
            shard = None
        with contextlib.ExitStack() as execution:
            locks = execution.enter_context(gpu_locks(shard['locks'])) if shard else []
            if shard:
                hard_limit = Timer(885, lambda: os.kill(os.getpid(), signal.SIGKILL))
                hard_limit.daemon = True
                hard_limit.start()
                execution.callback(hard_limit.cancel)
                execution.enter_context(shard_deadline(880))
                memory = execution.enter_context(process_memory_limit(shard['memory_reservation_bytes']))
                report['process_memory'] = memory
            import pytest, torch
            if shard:
                if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                    raise ValueError('The single reviewed visible CUDA device is unavailable')
                free, total = torch.cuda.mem_get_info(0)
                if free <= shard['memory_reservation_bytes']:
                    raise ValueError('Insufficient free GPU-1 memory for the shard reservation')
                torch.cuda.set_per_process_memory_fraction(shard['torch_allocator_limit_bytes'] / total, 0)
            build = dict(version=torch.__version__, git=torch.version.git_version, cuda=torch.version.cuda)
            if shard and build != catalog['torch_build']:
                raise ValueError('Torch build differs from collection')
            report.update(locks=locks, runtime=dict(torch_build=build, core_origin=str(core),
                core_distribution=importlib.metadata.version('witwin'),
                core_source_sha256=core_manifest['source_sha256'],
                frozen_core_inventory_sha256=core_manifest['frozen_inventory_sha256'],
                pythonpath=os.environ['PYTHONPATH'],
                physical_gpu=1 if shard else None, logical_device=0 if shard else None))

            class Evidence:
                def __init__(self):
                    self.nodes, self.rows, self.phases, self.tests = [], [], [], []
                    self.collection, self.warnings, self.deselected = [], [], []
                    self.active = None

                @pytest.hookimpl(hookwrapper=True)
                def pytest_make_collect_report(self, collector):
                    module = str(collector.path.relative_to(root)) if collector.path.is_relative_to(root) else ''
                    with module_environment(policy, module):
                        yield

                def pytest_collection_finish(self, session):
                    self.nodes = [item.nodeid for item in session.items]
                    self.rows = [dict(nodeid=item.nodeid, module=str(item.path.relative_to(root)),
                        line=item.location[1] + 1, fixtures=item.fixturenames,
                        skip_reasons=[str(m.kwargs.get('reason', '')) for m in item.iter_markers()
                                      if m.name in ('skip', 'skipif')]) for item in session.items]
                    self.binding = bound.verify()
                    self.benchmark_binding = benchmark.verify()
                    self.initial_ids = {n: id(obj) for n, (obj, _) in bound.modules.items()}
                    self.test_modules = {}
                    for name, obj in sys.modules.copy().items():
                        origin = getattr(obj, '__file__', None)
                        if origin and Path(origin).resolve().is_relative_to(root):
                            path = Path(origin).resolve()
                            if path.is_relative_to(root / 'tests') or path.is_relative_to(root / 'scripts'):
                                if str(path) not in before or file_record(path) != before[str(path)]:
                                    raise ValueError('Test/helper import escaped the frozen source inventory: ' + str(path))
                                self.test_modules[name] = (obj, str(path))
                    if shard and self.nodes != shard['nodeids']:
                        pytest.exit('Shard collection differs from exact assigned nodes', returncode=2)

                def pytest_collectreport(self, report):
                    self.collection.append(dict(nodeid=report.nodeid, outcome=report.outcome,
                        longrepr=str(report.longrepr) if report.failed else None))

                def pytest_deselected(self, items):
                    self.deselected.extend(item.nodeid for item in items)

                def pytest_warning_recorded(self, warning_message, when, nodeid, location):
                    self.warnings.append(dict(message=str(warning_message.message), when=when, nodeid=nodeid))

                def pytest_runtest_logreport(self, report):
                    self.phases.append(dict(nodeid=report.nodeid, when=report.when, outcome=report.outcome,
                        duration=report.duration, wasxfail=getattr(report, 'wasxfail', None),
                        longrepr=str(report.longrepr) if report.failed or report.skipped else None))

                @pytest.hookimpl(hookwrapper=True)
                def pytest_runtest_protocol(self, item, nextitem):
                    node_started = time.monotonic()
                    self.active = dict(nodeid=item.nodeid, cuda={}, measurement_errors=[])
                    with module_environment(policy, str(item.path.relative_to(root))):
                        torch.cuda.synchronize(0)
                        torch.cuda.reset_peak_memory_stats(0)
                        counter = torch.cuda.memory_stats(0)['allocation.all.allocated']
                        allocated = torch.cuda.memory_allocated(0)
                        free_before = torch.cuda.mem_get_info(0)[0]
                        yield
                        torch.cuda.synchronize(0)
                        self.active['cuda']['0'] = dict(allocation_events=torch.cuda.memory_stats(0)['allocation.all.allocated'] - counter,
                            allocated_before=allocated, allocated_after=torch.cuda.memory_allocated(0),
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                            device_free_before=free_before, device_free_after=torch.cuda.mem_get_info(0)[0])
                        self.active.update(seconds=time.monotonic() - node_started,
                                           host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                        self.tests.append(self.active)
                        with (out / 'node-progress.jsonl').open('a') as progress:
                            progress.write(json.dumps(self.active, sort_keys=True, allow_nan=False) + '\n')
                        self.active = None

            evidence = Evidence()
            command = (shard['nodeids'] if shard else SCOPE) + [
                '--gpu', '-q', '-p', 'no:cacheprovider', f'--junitxml={out}/junit.xml']
            if not shard:
                command.append('--collect-only')
            report['pytest_args'] = command
            with (out / 'pytest.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                code = int(pytest.main(command, plugins=[evidence]))
            report.update(pytest_exitstatus=code, ordered_nodeids=evidence.nodes, phases=evidence.phases,
                per_test=evidence.tests, collection=evidence.collection, warnings=evidence.warnings,
                deselected=evidence.deselected, source_inventory_sha256=canonical(logical))
            after = inventory(root, core, policy, policy_path, costs_path)
            final = bound.verify()
            write(out / 'source-after.json', after)
            write(out / 'initial-collection-bindings.json', evidence.binding)
            write(out / 'final-bindings.json', final)
            write(out / 'benchmark-collection-bindings.json', evidence.benchmark_binding)
            write(out / 'benchmark-final-bindings.json', benchmark.verify())
            write(out / 'test-helper-bindings.json', {n: dict(origin=p, sha256=before[p]['sha256'])
                  for n, (_, p) in evidence.test_modules.items()})
            if any(r['outcome'] != 'passed' for r in evidence.collection):
                raise ValueError('Acceptance collection failed; see the original collection reports and pytest.log')
            if (before != after or verify_source(root) != pair or evidence.deselected
                    or any(id(bound.modules[n][0]) != i for n, i in evidence.initial_ids.items())
                    or any(sys.modules.get(n) is not obj for n, (obj, _) in evidence.test_modules.items())):
                raise ValueError('Source/module binding or complete collection changed')
            report['junit_verification'] = verify_junit(out / 'junit.xml', evidence.nodes if shard else [])
            if not shard:
                if code != 0 or evidence.phases or evidence.tests or torch.cuda.is_initialized():
                    raise ValueError('Collection must be complete and remain host-only')
                admitted, multi, hash_only = partition(evidence.nodes)
                required = [r['nodeid'] for r in evidence.rows if r['module'] in (
                    'tests/sie/test_direct_restore.py', 'tests/sie/test_combined_direct_storage.py')
                    and 'CUDA required' in r['skip_reasons']]
                if len(required) != 22:
                    raise ValueError('The 22 required direct-storage allocation witnesses changed')
                shards = make_shards(admitted, costs)
                catalog = dict(protocol=PROTOCOL, scope=SCOPE, source_commit=SOURCE_COMMIT,
                    source_origin=str(root), source_inventory_sha256=canonical(logical),
                    harness_sha256=sha(Path(__file__).read_bytes()), torch_build=build,
                    full_nodeids=evidence.nodes, ordered_nodeids=admitted,
                    unqualified_multi_device_nodes=multi, hash_only_backend_nodes=hash_only,
                    unqualified_artifact_nodes=ARTIFACT_UNAVAILABLE_NODES,
                    required_cuda_nodeids=required, shards=shards, H2_G0='not_assessed',
                    unqualified_families=UNQUALIFIED_FAMILIES,
                    frozen_core=dict(root=core_manifest['root'], version=core_manifest['version'],
                        source_sha256=core_manifest['source_sha256'],
                        inventory_sha256=core_manifest['frozen_inventory_sha256']),
                    artifact_audit=audit, user_gpu_budget_seconds=28800,
                    estimated_gpu_seconds=sum(s['estimated_seconds'] for s in shards),
                    cost_model='1.5x historical phases plus 30s per shard; 60s per unmeasured node; no measured v3 costs')
                check_catalog(catalog, before, root, core)
                write(out / 'full-catalog.json', dict(protocol=PROTOCOL, nodes=evidence.rows,
                    source_inventory_sha256=canonical(logical), execution_pending=True))
                write(out / 'admitted-catalog.json', catalog)
                report.update(status='collected_no_tests_run', full_nodes=len(evidence.nodes),
                              admitted_nodes=len(admitted), cuda_initialized=False,
                              catalog_sha256=sha((out / 'admitted-catalog.json').read_bytes()))
            else:
                required = [n for n in catalog['required_cuda_nodeids'] if n in evidence.nodes]
                verdict = grade(evidence.nodes, evidence.phases, evidence.tests, code, True, ['0'], required)
                if evidence.warnings:
                    verdict['errors'].append('Pytest warnings remain')
                    verdict['passed'] = False
                report.update(verdict=verdict, engineering_acceptance_passed=verdict['passed'],
                              status='passed' if verdict['passed'] else 'failed')
    except BaseException as error:
        report['failure'] = dict(type=type(error).__name__, message=str(error))
    finally:
        if evidence is not None:
            report.update(ordered_nodeids=evidence.nodes, phases=evidence.phases, per_test=evidence.tests,
                          collection=evidence.collection, warnings=evidence.warnings, deselected=evidence.deselected,
                          incomplete_test_protocol=evidence.active)
            if args.mode == 'collect' and report['status'] == 'failed':
                write(out / 'incomplete-catalog.json', dict(status='collection_failed_not_executable',
                    protocol=PROTOCOL, scope=SCOPE, collected_nodeids=evidence.nodes,
                    collected_nodes=evidence.rows, collection_reports=evidence.collection,
                    source_inventory_sha256=canonical(logical), source_commit=SOURCE_COMMIT,
                    harness_sha256=sha(Path(__file__).read_bytes()), cuda_initialized=torch.cuda.is_initialized()))
        report['elapsed_seconds'] = time.monotonic() - started
        report['artifacts'] = {str(p.relative_to(out)): file_record(p) for p in out.iterdir() if p.is_file()}
        write(out / 'execution.json', report)
    print(json.dumps({k: report.get(k) for k in ('status', 'full_nodes', 'admitted_nodes', 'catalog_sha256', 'failure')}))
    return 0 if report['status'] in ('collected_no_tests_run', 'passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
