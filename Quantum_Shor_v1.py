# Quantum-8-Shor



import csv
import hashlib
import itertools
import math
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import qiskit
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector

random.seed(39)
np.random.seed(39)

FILES = (
    Path("/data/loto7_4716_k89.csv"),
    Path("/data/loto7_4716_k89_loto_2980.csv"),
    Path("/data/loto7_4716_k89_loto_plus_1736.csv"),
)

TITLE = "Quantum Shor - Modular Orbit"

N = 39
K = 7

SYSTEM_QUBITS = 6
DIM = 64

MODULUS = 41
BASES = (2, 3, 6, 7)

COUNTING_QUBITS = 12
REFERENCE_COUNTING_QUBITS = 4

COEFFICIENT_COUNTS = (4, 8)
REGULARIZATIONS = (0.001, 0.01)

TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

MAX_ITERATIONS = 200
GRADIENT_TOLERANCE = 1e-6

# SHOROV DEO
#
# Orakl je modularna eksponencijacija:
# f_a(x) = a**x mod modulus.
#
# QPE pocinje radnim stanjem |1>.
# Inverzna QFT daje raspodelu procenjenih faza.
# Verizni razlomci daju kandidate za red.
# Modularna provera i redukcija potvrđuju minimalni red.
#
# Puna QPE simulacija racuna sve amplitude vektorizovano.
# Manje QPE kolo nezavisno se proverava Qiskitom.
# Nema slucajnog merenja ili izbora faze.
#
# PREDIKTIVNA DOPUNA
#
# Brojevi se krecu po modularnim orbitama:
# next = previous * a**step mod 41.
#
# Koriste se baze sa punom orbitom nenultih ostataka.
# Ovaj izbor omogucava povezivanje svih brojeva 1..39.
#
# Amplitude koraka predstavljene su superpozicijom
# periodicnih sopstvenih stanja modularnog operatora.
# Njihovi kompleksni koeficijenti uce se iz istorije.
#
# Ovo je dodatna pretpostavka za Loto.
# Shorov red je aritmeticki red modularne funkcije.
#
# Broj 40 je pomocno modularno stanje.
# Zajednicka izlazna raspodela normalizuje se nad
# sedam razlicitih brojeva iz opsega 1..39.
#
# Zavrsna obuka koristi sve istorijske prelaze.
# Test ne bira hiperparametre.
#
# Program ne zapisuje fajlove i ne instalira pakete.


class DistributionError(ArithmeticError):
    pass


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_csv(path):
    raw = path.read_bytes()
    rows = []

    reader = csv.reader(
        raw.decode("utf-8-sig").splitlines()
    )

    for line_number, row in enumerate(reader, start=1):
        try:
            values = tuple(int(value.strip()) for value in row)
        except ValueError as error:
            raise ValueError(
                f"{path.name}: necelobrojan red {line_number}."
            ) from error

        check(
            len(values) == K
            and len(set(values)) == K
            and min(values) >= 1
            and max(values) <= N
            and values == tuple(sorted(values)),
            f"{path.name}: nelegalan red {line_number}.",
        )
        rows.append(values)

    check(
        len(rows) >= 5,
        f"{path.name}: nedovoljno kola.",
    )

    return {
        "path": path,
        "data": np.asarray(rows, dtype=np.int64) - 1,
        "digest": hashlib.sha256(raw).hexdigest(),
    }


def modular_multiplier(multiplier, modulus):
    check(
        1 < modulus <= DIM
        and math.gcd(multiplier, modulus) == 1,
        "Neispravna modularna multiplikacija.",
    )

    permutation = np.arange(DIM)
    permutation[:modulus] = (
        multiplier * np.arange(modulus)
    ) % modulus

    check(
        len(np.unique(permutation)) == DIM,
        "Modularni operator nije permutacija.",
    )

    operator = np.zeros(
        (DIM, DIM),
        dtype=complex,
    )
    operator[
        permutation,
        np.arange(DIM),
    ] = 1.0

    return operator


def controlled_multiplier(base, power, modulus):
    multiplier = pow(
        int(base),
        int(power),
        int(modulus),
    )
    work_operator = modular_multiplier(
        multiplier,
        modulus,
    )

    # Kontrolni kubit je prvi u qargs listi,
    # pa predstavlja najmanje znacajan bit ove matrice.
    operator = np.eye(
        2 * DIM,
        dtype=complex,
    )
    odd = np.arange(1, 2 * DIM, 2)

    operator[np.ix_(odd, odd)] = work_operator
    return operator


def qpe_state(base, modulus, counting_qubits):
    check(
        1 < base < modulus <= DIM
        and math.gcd(base, modulus) == 1,
        "Neispravni parametri QPE.",
    )

    size = 1 << counting_qubits

    residues = np.fromiter(
        (
            pow(int(base), exponent, int(modulus))
            for exponent in range(size)
        ),
        dtype=np.int64,
        count=size,
    )

    # Brojacki kubiti su najmanje znacajni.
    # Matrica ima oblik [radni ostatak, brojacko stanje].
    state = np.zeros(
        (DIM, size),
        dtype=complex,
    )
    state[
        residues,
        np.arange(size),
    ] = 1.0 / np.sqrt(size)

    # Inverzna QFT nad brojackim registrom.
    state = np.fft.fft(
        state,
        axis=1,
    ) / np.sqrt(size)

    check(
        np.all(np.isfinite(state))
        and np.isclose(
            np.sum(np.abs(state)**2),
            1.0,
            atol=1e-10,
            rtol=0,
        ),
        "QPE stanje nije normalizovano.",
    )

    return state


def verify_qpe_qiskit(base, modulus):
    counting = REFERENCE_COUNTING_QUBITS
    circuit = QuantumCircuit(
        counting + SYSTEM_QUBITS
    )

    circuit.h(range(counting))
    circuit.x(counting)

    work_qubits = list(range(
        counting,
        counting + SYSTEM_QUBITS,
    ))

    for control in range(counting):
        circuit.unitary(
            controlled_multiplier(
                base,
                1 << control,
                modulus,
            ),
            [control] + work_qubits,
        )

    circuit.append(
        QFT(
            counting,
            do_swaps=True,
        ).inverse(),
        range(counting),
    )

    actual = np.asarray(
        Statevector.from_instruction(circuit).data
    )
    expected = qpe_state(
        base,
        modulus,
        counting,
    ).ravel()

    check(
        np.allclose(
            actual,
            expected,
            atol=2e-10,
            rtol=2e-10,
        ),
        "Qiskit QPE i vektorizovana QPE nisu usaglasene.",
    )


def prime_factors(value):
    factors = []
    divisor = 2

    while divisor * divisor <= value:
        if value % divisor == 0:
            factors.append(divisor)

            while value % divisor == 0:
                value //= divisor

        divisor += 1

    if value > 1:
        factors.append(value)

    return factors


def reduce_order(base, modulus, candidate):
    check(
        pow(base, candidate, modulus) == 1,
        "Kandidat nije red modularnog operatora.",
    )

    order = candidate

    for factor in prime_factors(candidate):
        while (
            order % factor == 0
            and pow(base, order // factor, modulus) == 1
        ):
            order //= factor

    check(
        pow(base, order, modulus) == 1
        and all(
            pow(base, order // factor, modulus) != 1
            for factor in prime_factors(order)
        ),
        "Minimalni red nije potvrdjen.",
    )

    return order


def find_order(base, modulus):
    state = qpe_state(
        base,
        modulus,
        COUNTING_QUBITS,
    )
    probabilities = np.sum(
        np.abs(state)**2,
        axis=0,
    )
    size = len(probabilities)

    outcomes = np.argsort(
        -probabilities,
        kind="stable",
    )

    for outcome in outcomes:
        outcome = int(outcome)

        if outcome == 0 or probabilities[outcome] <= 0.0:
            continue

        fraction = Fraction(
            outcome,
            size,
        ).limit_denominator(modulus - 1)

        candidate = int(fraction.denominator)

        if (
            candidate > 1
            and pow(base, candidate, modulus) == 1
        ):
            order = reduce_order(
                base,
                modulus,
                candidate,
            )

            return {
                "order": order,
                "outcome": outcome,
                "fraction": str(fraction),
                "phase_probability": float(
                    probabilities[outcome]
                ),
            }

    raise RuntimeError(
        f"QPE nije dala potvrdjen red za bazu {base} "
        f"i modul {modulus}."
    )


def factors_from_order(base, modulus, order):
    if order % 2:
        return None

    root = pow(base, order // 2, modulus)
    first = math.gcd(root - 1, modulus)
    second = math.gcd(root + 1, modulus)

    if (
        1 < first < modulus
        and 1 < second < modulus
        and first * second == modulus
    ):
        return tuple(sorted((first, second)))

    return None


def build_order_info(base):
    verify_qpe_qiskit(base, MODULUS)
    certificate = find_order(base, MODULUS)
    order = certificate["order"]

    powers = np.array(
        [
            pow(base, step, MODULUS)
            for step in range(order)
        ],
        dtype=np.int64,
    )

    check(
        len(np.unique(powers)) == order
        and powers[0] == 1
        and pow(base, order, MODULUS) == 1,
        "Neispravna modularna orbita.",
    )

    lookup = np.full(
        MODULUS,
        -1,
        dtype=np.int64,
    )
    lookup[powers] = np.arange(order)

    positions = np.arange(order)
    fourier_block = np.exp(
        2j
        * np.pi
        * positions[:, None]
        * positions[None, :]
        / order
    ) / np.sqrt(order)

    fourier = np.eye(DIM, dtype=complex)
    fourier[:order, :order] = fourier_block

    check(
        np.allclose(
            fourier.conj().T @ fourier,
            np.eye(DIM),
            atol=2e-10,
            rtol=0,
        ),
        "Periodicka baza nije unitarna.",
    )

    print()
    print("Baza:", base)
    print("Modul:", MODULUS)
    print("QPE ishod:", certificate["outcome"])
    print("Verizni razlomak:", certificate["fraction"])
    print("Potvrdjeni aritmeticki red:", order)

    return {
        "base": base,
        "order": order,
        "powers": powers,
        "lookup": lookup,
        "fourier": fourier,
    }


def periodic_basis(info, coefficient_count):
    check(
        2 <= coefficient_count <= info["order"],
        "Neispravan broj periodicnih koeficijenata.",
    )

    return info["fourier"][
        :info["order"],
        :coefficient_count,
    ]


def initial_parameters(coefficient_count):
    return 0.10 * np.sin(
        np.arange(
            2 * (coefficient_count - 1),
            dtype=float,
        ) + 1.0
    )


def step_state(theta, basis):
    coefficient_count = basis.shape[1]
    free_count = coefficient_count - 1

    check(
        theta.shape == (2 * free_count,),
        "Neispravna dimenzija koeficijenata.",
    )

    coefficients = np.empty(
        coefficient_count,
        dtype=complex,
    )
    coefficients[0] = 1.0
    coefficients[1:] = (
        theta[:free_count]
        + 1j * theta[free_count:]
    )

    norm = float(np.linalg.norm(coefficients))
    amplitudes = basis @ coefficients / norm

    check(
        np.all(np.isfinite(amplitudes))
        and np.isclose(
            np.vdot(amplitudes, amplitudes).real,
            1.0,
            atol=1e-10,
            rtol=0,
        ),
        "Periodicke amplitude nisu normalizovane.",
    )

    return amplitudes, coefficients, norm


def log_amplitudes(amplitudes):
    magnitudes = np.abs(amplitudes)
    result = np.full(
        magnitudes.shape,
        -np.inf,
    )

    np.log(
        magnitudes,
        out=result,
        where=magnitudes > 0.0,
    )

    return 2.0 * result


def transition_counts(data, end, info):
    check(
        1 <= end < len(data),
        "Neispravan broj istorijskih prelaza.",
    )

    inverses = np.array(
        [
            pow(value, -1, MODULUS)
            for value in range(1, N + 1)
        ],
        dtype=np.int64,
    )

    ratios = (
        (data[1:end + 1] + 1)
        * inverses[data[:end]]
    ) % MODULUS

    steps = info["lookup"][ratios]

    check(
        np.all(steps >= 0),
        "Istorijski prelaz nije obuhvacen modularnom orbitom.",
    )

    counts = np.zeros(
        (K, info["order"]),
        dtype=float,
    )

    for slot in range(K):
        np.add.at(
            counts[slot],
            steps[:, slot],
            1.0,
        )

    check(
        np.all(counts.sum(axis=1) == end),
        "Nisu obuhvaceni svi istorijski prelazi.",
    )

    return counts


def objective(theta, counts, basis, weight):
    amplitudes, coefficients, norm = step_state(
        theta,
        basis,
    )

    total = float(counts.sum())
    check(total > 0.0, "Prazan trening skup.")

    frequencies = counts / total
    probabilities = np.abs(amplitudes)**2

    # Brierov gubitak nad celom raspodelom modularnih koraka.
    loss = (
        np.dot(probabilities, probabilities)
        - 2.0 * np.dot(frequencies, probabilities)
        + 1.0
    )

    probability_derivative = (
        2.0 * (probabilities - frequencies)
    )
    amplitude_adjoint = (
        2.0 * probability_derivative * amplitudes
    )

    coefficient_gradient = (
        basis.conj().T @ amplitude_adjoint / norm
        - coefficients
        * float(np.real(np.vdot(
            amplitude_adjoint,
            amplitudes,
        )))
        / norm**2
    )

    gradient = np.concatenate((
        coefficient_gradient[1:].real,
        coefficient_gradient[1:].imag,
    ))

    loss += weight * np.dot(theta, theta)
    gradient += 2.0 * weight * theta

    check(
        np.isfinite(loss)
        and np.all(np.isfinite(gradient)),
        "Nekonacan gubitak ili gradijent.",
    )

    return float(loss), gradient


def fit_models(counts, basis, weight, label):
    print(label)
    print("Istorijskih prelaza:", int(counts[0].sum()))

    models = []
    coefficient_count = basis.shape[1]

    for slot in range(K):
        starting_point = initial_parameters(
            coefficient_count
        )
        best_loss = np.inf
        best_theta = starting_point.copy()

        def tracked_objective(candidate):
            nonlocal best_loss, best_theta

            loss, gradient = objective(
                candidate,
                counts[slot],
                basis,
                weight,
            )

            if loss < best_loss:
                best_loss = loss
                best_theta = candidate.copy()

            return loss, gradient

        result = minimize(
            tracked_objective,
            starting_point,
            method="CG",
            jac=True,
            options={
                "maxiter": MAX_ITERATIONS,
                "gtol": GRADIENT_TOLERANCE,
            },
        )

        loss, gradient = objective(
            best_theta,
            counts[slot],
            basis,
            weight,
        )
        gradient_norm = float(
            np.max(np.abs(gradient))
        )

        check(
            np.isfinite(best_loss)
            and np.all(np.isfinite(best_theta)),
            "Optimizator nije sacuvao konacne parametre.",
        )

        print(
            "Registar:",
            slot + 1,
            "gubitak:",
            f"{loss:.8f}",
            "norma gradijenta:",
            f"{gradient_norm:.8g}",
            flush=True,
        )
        print("Status optimizatora:", result.message)

        if (
            not result.success
            or gradient_norm > GRADIENT_TOLERANCE
        ):
            print(
                "UPOZORENJE: konvergencija nije potvrdjena."
            )

        models.append(best_theta.copy())

    return models


def orbit_indices(info, previous):
    check(
        1 <= previous <= N,
        "Neispravno prethodno stanje.",
    )

    destinations = (
        previous * info["powers"]
    ) % MODULUS - 1

    check(
        len(np.unique(destinations)) == info["order"]
        and np.all((destinations >= 0) & (destinations < DIM)),
        "Neispravni indeksi modularne orbite.",
    )

    return destinations


def orbit_operator(info, previous):
    destinations = orbit_indices(info, previous)
    used = set(destinations.tolist())

    remaining = np.array(
        [
            value
            for value in range(DIM)
            if value not in used
        ],
        dtype=np.int64,
    )

    permutation = np.concatenate((
        destinations,
        remaining,
    ))

    check(
        len(np.unique(permutation)) == DIM,
        "Modularno preslikavanje nije permutacija.",
    )

    operator = np.zeros(
        (DIM, DIM),
        dtype=complex,
    )
    operator[
        permutation,
        np.arange(DIM),
    ] = 1.0

    return operator


def make_log_kernels(models, basis, info):
    check(
        len(models) == K,
        "Nedostaju modeli registara.",
    )

    kernels = []

    for theta in models:
        amplitudes, _, _ = step_state(theta, basis)
        step_logs = log_amplitudes(amplitudes)

        kernel = np.full(
            (DIM, N),
            -np.inf,
        )

        for previous in range(1, N + 1):
            kernel[
                orbit_indices(info, previous),
                previous - 1,
            ] = step_logs

        check(
            np.allclose(
                logsumexp(kernel, axis=0),
                0.0,
                atol=1e-10,
                rtol=0,
            ),
            "Modularna raspodela nije normalizovana.",
        )

        kernels.append(kernel)

    return kernels


def joint_distribution(log_weights):
    logs = np.asarray(log_weights, dtype=float).copy()

    check(
        logs.ndim == 2
        and np.all(np.isfinite(logs) | np.isneginf(logs)),
        "Neispravni logaritmi zajednicke raspodele.",
    )

    slots, states = logs.shape
    check(
        1 <= slots <= states,
        "Neispravne dimenzije raspodele.",
    )

    indices = np.arange(states)
    increasing = indices[:, None] < indices[None, :]

    allowed = (
        (indices[None, :] >= np.arange(slots)[:, None])
        & (
            indices[None, :]
            <= states - slots + np.arange(slots)[:, None]
        )
    )
    logs[~allowed] = -np.inf

    forward = np.full_like(logs, -np.inf)
    backward = np.full_like(logs, -np.inf)
    forward[0] = logs[0]

    maximum = logs[0].copy()
    parents = np.zeros(
        (slots, states),
        dtype=np.int64,
    )

    for slot in range(1, slots):
        forward[slot] = logs[slot] + logsumexp(
            np.where(
                increasing,
                forward[slot - 1, :, None],
                -np.inf,
            ),
            axis=0,
        )

        candidates = np.where(
            increasing,
            maximum[:, None],
            -np.inf,
        )
        parents[slot] = candidates.argmax(axis=0)
        maximum = logs[slot] + candidates[
            parents[slot],
            np.arange(states),
        ]

    log_partition = float(
        logsumexp(forward[-1])
    )

    if not np.isfinite(log_partition):
        raise DistributionError(
            "Modularno stanje nema pozitivnu legalnu masu."
        )

    path = np.zeros(slots, dtype=np.int64)
    path[-1] = int(maximum.argmax())

    for slot in range(slots - 1, 0, -1):
        path[slot - 1] = parents[slot, path[slot]]

    backward[-1] = 0.0

    for slot in range(slots - 2, -1, -1):
        backward[slot] = logsumexp(
            np.where(
                increasing,
                (
                    logs[slot + 1]
                    + backward[slot + 1]
                )[None, :],
                -np.inf,
            ),
            axis=1,
        )

    marginals = np.exp(
        forward + backward - log_partition
    )
    mode_log_mass = float(
        logs[np.arange(slots), path].sum()
    )

    check(
        np.all(np.diff(path) > 0)
        and np.isfinite(mode_log_mass)
        and mode_log_mass <= log_partition + 1e-8
        and log_partition <= 1e-8,
        "Neispravan globalni maksimum.",
    )
    check(
        np.allclose(
            marginals.sum(axis=1),
            1.0,
            atol=1e-8,
            rtol=0,
        ),
        "Zajednicka raspodela nije normalizovana.",
    )

    return {
        "path": path,
        "marginals": marginals,
        "log_partition": log_partition,
        "log_mode_probability": (
            mode_log_mass - log_partition
        ),
        "logs": logs,
    }


def evaluate(models, basis, info, data, times):
    kernels = make_log_kernels(
        models,
        basis,
        info,
    )

    losses = []
    hits = []

    for time in times:
        logs = np.array([
            kernels[slot][
                :N,
                data[time - 1, slot],
            ]
            for slot in range(K)
        ])

        distribution = joint_distribution(logs)

        target_log_mass = float(
            distribution["logs"][
                np.arange(K),
                data[time],
            ].sum()
        )

        loss = (
            distribution["log_partition"]
            - target_log_mass
        )

        check(
            not np.isnan(loss) and loss >= -1e-8,
            "Neispravan evaluacioni NLL.",
        )

        losses.append(loss)
        hits.append(
            len(
                set(distribution["path"])
                & set(data[time])
            )
        )

    check(len(hits) > 0, "Prazan evaluacioni skup.")
    losses = np.asarray(losses, dtype=float)

    return {
        "nll": float(losses.mean()),
        "hits": np.asarray(hits, dtype=np.int64),
        "zero_targets": int(
            np.isposinf(losses).sum()
        ),
    }


def verify_forecast_qiskit(theta, basis, info, previous):
    amplitudes, coefficients, norm = step_state(
        theta,
        basis,
    )

    initial_state = np.zeros(
        DIM,
        dtype=complex,
    )
    initial_state[:len(coefficients)] = coefficients / norm

    mapping = orbit_operator(info, previous)

    circuit = QuantumCircuit(SYSTEM_QUBITS)
    circuit.initialize(
        initial_state,
        range(SYSTEM_QUBITS),
    )
    circuit.unitary(
        info["fourier"],
        range(SYSTEM_QUBITS),
    )
    circuit.unitary(
        mapping,
        range(SYSTEM_QUBITS),
    )

    actual = np.asarray(
        Statevector.from_instruction(circuit).data
    )

    expected = np.zeros(
        DIM,
        dtype=complex,
    )
    expected[
        orbit_indices(info, previous)
    ] = amplitudes

    check(
        np.allclose(
            actual,
            expected,
            atol=2e-10,
            rtol=2e-10,
        ),
        "Qiskit i modularne prediktivne amplitude nisu usaglasene.",
    )


def self_checks(data, infos):
    verify_qpe_qiskit(7, 15)

    certificate = find_order(7, 15)
    factors = factors_from_order(
        7,
        15,
        certificate["order"],
    )

    check(
        factors is not None
        and factors[0] * factors[1] == 15,
        "Shorova provera faktorizacije nije prosla.",
    )

    print()
    print("Shorova provera faktorizacije 15:", factors)

    info = infos[0]
    counts = transition_counts(
        data,
        min(8, len(data) - 1),
        info,
    )[0]

    for coefficient_count in COEFFICIENT_COUNTS:
        basis = periodic_basis(
            info,
            coefficient_count,
        )
        theta = initial_parameters(
            coefficient_count
        )

        verify_forecast_qiskit(
            theta,
            basis,
            info,
            int(data[0, 0] + 1),
        )

        _, gradient = objective(
            theta,
            counts,
            basis,
            0.001,
        )

        for index in range(len(theta)):
            offset = np.zeros_like(theta)
            offset[index] = 1e-6

            numerical = (
                objective(
                    theta + offset,
                    counts,
                    basis,
                    0.001,
                )[0]
                - objective(
                    theta - offset,
                    counts,
                    basis,
                    0.001,
                )[0]
            ) / 2e-6

            check(
                np.isclose(
                    numerical,
                    gradient[index],
                    atol=2e-6,
                    rtol=2e-4,
                ),
                f"Gradijent nije potvrdjen za "
                f"{coefficient_count} koeficijenata, "
                f"parametar {index}.",
            )

    toy = softmax(
        np.sin(np.arange(15).reshape(3, 5)),
        axis=1,
    )
    distribution = joint_distribution(np.log(toy))

    scores = [
        sum(
            np.log(toy[slot, combination[slot]])
            for slot in range(3)
        )
        for combination in itertools.combinations(range(5), 3)
    ]

    path = distribution["path"]
    mode_score = sum(
        np.log(toy[slot, path[slot]])
        for slot in range(3)
    )

    check(
        np.isclose(
            distribution["log_partition"],
            logsumexp(scores),
            atol=1e-10,
            rtol=0,
        )
        and np.isclose(
            mode_score,
            max(scores),
            atol=1e-10,
            rtol=0,
        ),
        "Provera normalizacije i globalnog maksimuma nije prosla.",
    )

    tiny_log = log_amplitudes(
        np.array([1e-180 + 0j])
    )[0]

    check(
        np.isfinite(tiny_log),
        "Provera male amplitude nije prosla.",
    )

    print("Ugradjene pocetne provere su prosle.")


def print_metrics(label, metrics):
    print(label)
    print("Broj kola:", len(metrics["hits"]))
    print("Srednji NLL:", f'{metrics["nll"]:.8f}')
    print(
        "Srednji broj pogodaka:",
        f'{metrics["hits"].mean():.6f}',
    )
    print(
        "Nulte ciljne verovatnoce:",
        metrics["zero_targets"],
    )
    print(
        "Broj kola po pogocima od 0 do 7:",
        np.bincount(
            metrics["hits"],
            minlength=K + 1,
        ).tolist(),
    )


def process_dataset(dataset, infos):
    path = dataset["path"]
    data = dataset["data"]
    total = len(data) - 1

    train_end = int(TRAIN_FRACTION * total)
    validation_end = int(
        VALIDATION_END_FRACTION * total
    )

    check(
        0 < train_end < validation_end < total,
        "Neispravna hronoloska podela.",
    )

    print()
    print("CSV:", path.name)
    print("Redova:", len(data))
    print("SHA256:", dataset["digest"])
    print("Trening prelaza:", train_end)
    print(
        "Validacionih kola:",
        validation_end - train_end,
    )
    print("Test kola:", total - validation_end)

    best_settings = None
    best_validation = np.inf

    for info in infos:
        counts = transition_counts(
            data,
            train_end,
            info,
        )

        for coefficient_count in COEFFICIENT_COUNTS:
            basis = periodic_basis(
                info,
                coefficient_count,
            )

            for weight in REGULARIZATIONS:
                print()
                print(
                    "Kandidat:",
                    "baza",
                    info["base"],
                    "red",
                    info["order"],
                    "koeficijenata",
                    coefficient_count,
                    "regularizacija",
                    weight,
                )

                models = fit_models(
                    counts,
                    basis,
                    weight,
                    "UCENJE ZA VALIDACIJU",
                )

                try:
                    metrics = evaluate(
                        models,
                        basis,
                        info,
                        data,
                        range(
                            train_end + 1,
                            validation_end + 1,
                        ),
                    )
                except DistributionError as error:
                    print(
                        "Kandidat nema definisanu legalnu raspodelu."
                    )
                    print(str(error))
                    continue

                print_metrics(
                    "VALIDACIONI REZULTAT",
                    metrics,
                )

                if (
                    np.isfinite(metrics["nll"])
                    and metrics["nll"] < best_validation
                ):
                    best_validation = metrics["nll"]
                    best_settings = (
                        info,
                        coefficient_count,
                        weight,
                    )

    check(
        best_settings is not None,
        "Nijedan kandidat nema konacan validacioni NLL.",
    )

    info, coefficient_count, weight = best_settings
    basis = periodic_basis(
        info,
        coefficient_count,
    )

    print()
    print("Izabrana modularna baza:", info["base"])
    print("QPE aritmeticki red:", info["order"])
    print("Broj koeficijenata:", coefficient_count)
    print("Regularizacija:", weight)

    test_counts = transition_counts(
        data,
        validation_end,
        info,
    )
    test_models = fit_models(
        test_counts,
        basis,
        weight,
        "UCENJE ZA ODVOJENI TEST",
    )

    test_metrics = evaluate(
        test_models,
        basis,
        info,
        data,
        range(validation_end + 1, len(data)),
    )
    print_metrics(
        "ODVOJENI TEST",
        test_metrics,
    )

    # Model ostaje zamrznut tokom testa.
    # Svako test kolo koristi samo prethodno poznato kolo.
    # Test rezultat ne bira hiperparametre.

    final_counts = transition_counts(
        data,
        total,
        info,
    )
    final_models = fit_models(
        final_counts,
        basis,
        weight,
        "ZAVRSNO UCENJE NA SVIM ISTORIJSKIM PRELAZIMA",
    )

    kernels = make_log_kernels(
        final_models,
        basis,
        info,
    )

    next_logs = []

    for slot, theta in enumerate(final_models):
        previous = int(data[-1, slot] + 1)

        verify_forecast_qiskit(
            theta,
            basis,
            info,
            previous,
        )
        next_logs.append(
            kernels[slot][:N, previous - 1]
        )

    print(
        "Zavrsne Qiskit provere svih sedam registara su prosle."
    )

    next_logs = np.asarray(next_logs)
    distribution = joint_distribution(next_logs)

    prediction = distribution["path"]
    combination = prediction + 1

    print(
        "Log legalne mase:",
        f'{distribution["log_partition"]:.8f}',
    )
    print(
        "Log modelske verovatnoce NEXT:",
        f'{distribution["log_mode_probability"]:.8f}',
    )

    if np.array_equal(prediction, data[-1]):
        print("PROVERA: NEXT ponavlja poslednje kolo.")

    longest_run = 1
    current_run = 1

    for difference in np.diff(combination):
        if difference == 1:
            current_run += 1
            longest_run = max(
                longest_run,
                current_run,
            )
        else:
            current_run = 1

    if longest_run >= 5:
        print(
            "PROVERA: najmanje pet uzastopnih brojeva."
        )

    span = int(combination[-1] - combination[0])
    historical_spans = data[:, -1] - data[:, 0]

    print("Raspon NEXT kombinacije:", span)
    print(
        "Udeo istorijskih raspona manjih ili jednakih NEXT rasponu:",
        f"{100.0 * np.mean(historical_spans <= span):.3f}%",
    )

    return {
        "name": path.name,
        "digest": dataset["digest"],
        "next": combination,
        "probabilities": np.exp(next_logs),
        "marginals": distribution["marginals"],
    }


def main():
    print(TITLE)
    print("Qiskit:", qiskit.__version__)
    print("Shorov podalgoritam nalazenja modularnog reda.")
    print("Modul za Loto adaptaciju:", MODULUS)
    print(
        "QPE pomocnih kubita:",
        COUNTING_QUBITS,
    )
    print(
        "QPE radnih kubita:",
        SYSTEM_QUBITS,
    )
    print(
        "Qiskit referentna QPE koristi pomocnih kubita:",
        REFERENCE_COUNTING_QUBITS,
    )
    print(
        "Prediktivna dopuna uci amplitude modularnih prelaza."
    )

    check(
        N < MODULUS <= DIM
        and all(
            math.gcd(value, MODULUS) == 1
            for value in range(1, N + 1)
        ),
        "Modul ne podrzava sve Loto brojeve kao invertibilna stanja.",
    )

    datasets = [
        load_csv(path)
        for path in FILES
    ]

    all_infos = [
        build_order_info(base)
        for base in BASES
    ]

    infos = []

    for info in all_infos:
        if info["order"] == MODULUS - 1:
            infos.append(info)
        else:
            print(
                "Baza",
                info["base"],
                "ima nepotpunu orbitu za ovaj prediktivni model.",
            )

    check(
        len(infos) > 0,
        "QPE nije potvrdila nijednu punu modularnu orbitu.",
    )

    self_checks(datasets[0]["data"], infos)

    results = [
        process_dataset(dataset, infos)
        for dataset in datasets
    ]

    for first, second in itertools.combinations(results, 2):
        if first["digest"] == second["digest"]:
            print("PROVERA: dva ulaza imaju isti SHA256.")
            print(first["name"])
            print(second["name"])

        if np.array_equal(
            first["next"],
            second["next"],
        ):
            print()
            print("PROVERA ISTIH NEXT KOMBINACIJA")
            print(first["name"])
            print(second["name"])
            print(
                "Najveca razlika sirovih raspodela:",
                float(np.max(np.abs(
                    first["probabilities"]
                    - second["probabilities"]
                ))),
            )
            print(
                "Najveca razlika legalnih marginalnih raspodela:",
                float(np.max(np.abs(
                    first["marginals"]
                    - second["marginals"]
                ))),
            )

    print()
    print("ZAVRSNE NEXT KOMBINACIJE")

    for result in results:
        print(result["name"])
        print(
            "NEXT:",
            " ".join(map(str, result["next"])),
        )


if __name__ == "__main__":
    main()






"""
Quantum Shor - Modular Orbit
Qiskit: 1.4.4
Shorov podalgoritam nalazenja modularnog reda.
Modul za Loto adaptaciju: 41
QPE pomocnih kubita: 12
QPE radnih kubita: 6
Qiskit referentna QPE koristi pomocnih kubita: 4
Prediktivna dopuna uci amplitude modularnih prelaza.

Baza: 2
Modul: 41
QPE ishod: 205
Verizni razlomak: 1/20
Potvrdjeni aritmeticki red: 20

Baza: 3
Modul: 41
QPE ishod: 512
Verizni razlomak: 1/8
Potvrdjeni aritmeticki red: 8

Baza: 6
Modul: 41
QPE ishod: 307
Verizni razlomak: 3/40
Potvrdjeni aritmeticki red: 40

Baza: 7
Modul: 41
QPE ishod: 307
Verizni razlomak: 3/40
Potvrdjeni aritmeticki red: 40
Baza 2 ima nepotpunu orbitu za ovaj prediktivni model.
Baza 3 ima nepotpunu orbitu za ovaj prediktivni model.

Shorova provera faktorizacije 15: (3, 5)
Ugradjene pocetne provere su prosle.

CSV: loto7_4716_k89.csv
Redova: 4716
SHA256: deb02ea741f500e3270402730ed89565f1c9b4618d53e5a08fea2b7c2c2af31a
Trening prelaza: 3300
Validacionih kola: 707
Test kola: 708

Kandidat: baza 6 red 40 koeficijenata 4 regularizacija 0.001
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 3300
Registar: 1 gubitak: 0.96893489 norma gradijenta: 8.4716042e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97470583 norma gradijenta: 9.8624379e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97488350 norma gradijenta: 2.9472167e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97491247 norma gradijenta: 1.1420762e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97488573 norma gradijenta: 3.812233e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97478314 norma gradijenta: 5.3359604e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97143539 norma gradijenta: 7.2608364e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 16.68992844
Srednji broj pogodaka: 1.258840
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [155, 294, 187, 63, 7, 1, 0, 0]

Kandidat: baza 6 red 40 koeficijenata 4 regularizacija 0.01
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 3300
Registar: 1 gubitak: 0.96997815 norma gradijenta: 9.2014636e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97475014 norma gradijenta: 5.178063e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97490052 norma gradijenta: 8.4701969e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97492563 norma gradijenta: 8.8314983e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97490253 norma gradijenta: 1.1283642e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97481670 norma gradijenta: 2.5297669e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97201425 norma gradijenta: 5.344448e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 16.63621014
Srednji broj pogodaka: 1.258840
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [155, 294, 187, 63, 7, 1, 0, 0]

Kandidat: baza 6 red 40 koeficijenata 8 regularizacija 0.001
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 3300
Registar: 1 gubitak: 0.96788303 norma gradijenta: 8.6639255e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97442442 norma gradijenta: 9.0528261e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97473325 norma gradijenta: 5.4221468e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97460530 norma gradijenta: 1.6373473e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97485079 norma gradijenta: 2.8444926e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97456933 norma gradijenta: 9.3743971e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97123011 norma gradijenta: 4.0696069e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 16.72919573
Srednji broj pogodaka: 1.239038
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [167, 285, 187, 56, 11, 1, 0, 0]

Kandidat: baza 6 red 40 koeficijenata 8 regularizacija 0.01
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 3300
Registar: 1 gubitak: 0.96894717 norma gradijenta: 4.8763405e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97451173 norma gradijenta: 4.2335669e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97477083 norma gradijenta: 1.3125755e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97466290 norma gradijenta: 2.2275706e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97487277 norma gradijenta: 6.7314873e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97463399 norma gradijenta: 5.840823e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97182990 norma gradijenta: 8.9027332e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 16.67724531
Srednji broj pogodaka: 1.239038
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [167, 285, 187, 56, 11, 1, 0, 0]

Kandidat: baza 7 red 40 koeficijenata 4 regularizacija 0.001
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 3300
Registar: 1 gubitak: 0.96893489 norma gradijenta: 9.1255333e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97470583 norma gradijenta: 1.281112e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97488350 norma gradijenta: 1.8814491e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97491247 norma gradijenta: 1.0816246e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97488573 norma gradijenta: 8.5179572e-08
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97478314 norma gradijenta: 1.901776e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97143539 norma gradijenta: 2.8994661e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 16.68992844
Srednji broj pogodaka: 1.258840
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [155, 294, 187, 63, 7, 1, 0, 0]

Kandidat: baza 7 red 40 koeficijenata 4 regularizacija 0.01
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 3300
Registar: 1 gubitak: 0.96997815 norma gradijenta: 7.32842e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97475014 norma gradijenta: 3.5672207e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97490052 norma gradijenta: 8.7237956e-08
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97492563 norma gradijenta: 8.2738412e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97490253 norma gradijenta: 1.3830864e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97481670 norma gradijenta: 7.6256638e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97201425 norma gradijenta: 4.723396e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 16.63621494
Srednji broj pogodaka: 1.258840
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [155, 294, 187, 63, 7, 1, 0, 0]

Kandidat: baza 7 red 40 koeficijenata 8 regularizacija 0.001
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 3300
Registar: 1 gubitak: 0.96788303 norma gradijenta: 9.769447e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97442442 norma gradijenta: 3.6312964e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97473325 norma gradijenta: 2.3637534e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97460530 norma gradijenta: 2.0320308e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97485079 norma gradijenta: 1.5431059e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97456933 norma gradijenta: 2.589672e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97123011 norma gradijenta: 7.0325083e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 16.72920394
Srednji broj pogodaka: 1.239038
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [167, 285, 187, 56, 11, 1, 0, 0]

Kandidat: baza 7 red 40 koeficijenata 8 regularizacija 0.01
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 3300
Registar: 1 gubitak: 0.96894717 norma gradijenta: 4.2438569e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97451173 norma gradijenta: 9.1788566e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97477083 norma gradijenta: 1.4150928e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97466290 norma gradijenta: 1.5652823e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97487277 norma gradijenta: 4.1747284e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97463399 norma gradijenta: 6.3591735e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97182990 norma gradijenta: 8.8206611e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 707
Srednji NLL: 16.67724756
Srednji broj pogodaka: 1.239038
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [167, 285, 187, 56, 11, 1, 0, 0]

Izabrana modularna baza: 6
QPE aritmeticki red: 40
Broj koeficijenata: 4
Regularizacija: 0.01
UCENJE ZA ODVOJENI TEST
Istorijskih prelaza: 4007
Registar: 1 gubitak: 0.97008744 norma gradijenta: 9.3893633e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97475281 norma gradijenta: 4.734037e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97491310 norma gradijenta: 8.9753922e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97493506 norma gradijenta: 5.5483901e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97488279 norma gradijenta: 7.6481898e-08
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97482880 norma gradijenta: 2.5977243e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97183009 norma gradijenta: 8.2305493e-07
Status optimizatora: Optimization terminated successfully.
ODVOJENI TEST
Broj kola: 708
Srednji NLL: 16.57809665
Srednji broj pogodaka: 1.240113
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [152, 304, 188, 58, 6, 0, 0, 0]
ZAVRSNO UCENJE NA SVIM ISTORIJSKIM PRELAZIMA
Istorijskih prelaza: 4715
Registar: 1 gubitak: 0.96979081 norma gradijenta: 4.2401611e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97471256 norma gradijenta: 5.2465671e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97490505 norma gradijenta: 8.0140166e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97492476 norma gradijenta: 9.9152653e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97489388 norma gradijenta: 7.4645865e-08
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97479140 norma gradijenta: 6.2910866e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97173502 norma gradijenta: 1.4272328e-07
Status optimizatora: Optimization terminated successfully.
Zavrsne Qiskit provere svih sedam registara su prosle.
Log legalne mase: -8.79483221
Log modelske verovatnoce NEXT: -15.53061225
Raspon NEXT kombinacije: 33
Udeo istorijskih raspona manjih ili jednakih NEXT rasponu: 71.904%

CSV: loto7_4716_k89_loto_2980.csv
Redova: 2979
SHA256: f1cd81cd04660d50d9058675924da34346fc006ce8dabd84d52266c66afe17d1
Trening prelaza: 2084
Validacionih kola: 447
Test kola: 447

Kandidat: baza 6 red 40 koeficijenata 4 regularizacija 0.001
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 2084
Registar: 1 gubitak: 0.96889014 norma gradijenta: 6.2852486e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97437136 norma gradijenta: 1.222297e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97472274 norma gradijenta: 6.5556495e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97486552 norma gradijenta: 4.2460547e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97489718 norma gradijenta: 2.4037702e-08
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97467174 norma gradijenta: 3.0439003e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97177655 norma gradijenta: 9.5916206e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 16.74069836
Srednji broj pogodaka: 1.246085
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [89, 206, 112, 33, 7, 0, 0, 0]

Kandidat: baza 6 red 40 koeficijenata 4 regularizacija 0.01
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 2084
Registar: 1 gubitak: 0.96992624 norma gradijenta: 8.56601e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97446957 norma gradijenta: 3.2080811e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97476483 norma gradijenta: 5.4882062e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97488596 norma gradijenta: 6.7161124e-08
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97491261 norma gradijenta: 4.564374e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97472390 norma gradijenta: 5.5155955e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97229439 norma gradijenta: 5.7621087e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 16.67676114
Srednji broj pogodaka: 1.261745
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [88, 207, 106, 39, 7, 0, 0, 0]

Kandidat: baza 6 red 40 koeficijenata 8 regularizacija 0.001
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 2084
Registar: 1 gubitak: 0.96789782 norma gradijenta: 5.8303719e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97384997 norma gradijenta: 2.57869e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97458976 norma gradijenta: 4.1838549e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97464577 norma gradijenta: 7.573544e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97483832 norma gradijenta: 5.8045579e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97432720 norma gradijenta: 4.4049126e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97144568 norma gradijenta: 5.8431418e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 16.76484555
Srednji broj pogodaka: 1.266219
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [94, 189, 121, 37, 6, 0, 0, 0]

Kandidat: baza 6 red 40 koeficijenata 8 regularizacija 0.01
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 2084
Registar: 1 gubitak: 0.96893656 norma gradijenta: 4.0099977e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97402294 norma gradijenta: 6.0779885e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97465027 norma gradijenta: 2.5429697e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97469806 norma gradijenta: 8.2998156e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97486269 norma gradijenta: 1.4847879e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97443262 norma gradijenta: 5.1918658e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97200806 norma gradijenta: 7.6119175e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 16.70579849
Srednji broj pogodaka: 1.266219
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [94, 189, 121, 37, 6, 0, 0, 0]

Kandidat: baza 7 red 40 koeficijenata 4 regularizacija 0.001
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 2084
Registar: 1 gubitak: 0.96889014 norma gradijenta: 5.5790512e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97437136 norma gradijenta: 3.1695091e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97472274 norma gradijenta: 5.4977224e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97486552 norma gradijenta: 8.0478262e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97489718 norma gradijenta: 7.4786125e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97467174 norma gradijenta: 7.4183063e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97177655 norma gradijenta: 5.5873686e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 16.74070207
Srednji broj pogodaka: 1.248322
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [89, 206, 111, 34, 7, 0, 0, 0]

Kandidat: baza 7 red 40 koeficijenata 4 regularizacija 0.01
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 2084
Registar: 1 gubitak: 0.96992624 norma gradijenta: 5.2942971e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97446957 norma gradijenta: 5.9785766e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97476483 norma gradijenta: 9.7898352e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97488596 norma gradijenta: 2.0046648e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97491261 norma gradijenta: 8.7726528e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97472390 norma gradijenta: 8.0739799e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97229439 norma gradijenta: 6.4614915e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 16.67675665
Srednji broj pogodaka: 1.261745
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [88, 207, 106, 39, 7, 0, 0, 0]

Kandidat: baza 7 red 40 koeficijenata 8 regularizacija 0.001
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 2084
Registar: 1 gubitak: 0.96789782 norma gradijenta: 9.4526925e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97384997 norma gradijenta: 3.9097509e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97458976 norma gradijenta: 5.024888e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97464577 norma gradijenta: 8.9461478e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97483832 norma gradijenta: 3.2163423e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97432720 norma gradijenta: 5.1580502e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97144568 norma gradijenta: 9.2625041e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 16.76484633
Srednji broj pogodaka: 1.266219
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [94, 189, 121, 37, 6, 0, 0, 0]

Kandidat: baza 7 red 40 koeficijenata 8 regularizacija 0.01
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 2084
Registar: 1 gubitak: 0.96893656 norma gradijenta: 4.4983787e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97402294 norma gradijenta: 8.116598e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97465027 norma gradijenta: 8.203045e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97469806 norma gradijenta: 2.129079e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97486269 norma gradijenta: 7.8512332e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97443262 norma gradijenta: 5.3158474e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97200806 norma gradijenta: 6.8325362e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 447
Srednji NLL: 16.70579906
Srednji broj pogodaka: 1.266219
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [94, 189, 121, 37, 6, 0, 0, 0]

Izabrana modularna baza: 7
QPE aritmeticki red: 40
Broj koeficijenata: 4
Regularizacija: 0.01
UCENJE ZA ODVOJENI TEST
Istorijskih prelaza: 2531
Registar: 1 gubitak: 0.97034219 norma gradijenta: 6.7187525e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97457091 norma gradijenta: 5.5019561e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97474864 norma gradijenta: 9.8949022e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97492080 norma gradijenta: 2.1920693e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97490854 norma gradijenta: 8.092128e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97477603 norma gradijenta: 8.3638485e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97219873 norma gradijenta: 5.7616108e-07
Status optimizatora: Optimization terminated successfully.
ODVOJENI TEST
Broj kola: 447
Srednji NLL: 16.64443617
Srednji broj pogodaka: 1.304251
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [89, 186, 127, 37, 8, 0, 0, 0]
ZAVRSNO UCENJE NA SVIM ISTORIJSKIM PRELAZIMA
Istorijskih prelaza: 2978
Registar: 1 gubitak: 0.97043755 norma gradijenta: 6.2065341e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97460360 norma gradijenta: 2.6665848e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97480838 norma gradijenta: 9.4475858e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97492016 norma gradijenta: 7.2765634e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97490710 norma gradijenta: 7.5392076e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97477416 norma gradijenta: 1.0368518e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97203897 norma gradijenta: 5.5142468e-07
Status optimizatora: Optimization terminated successfully.
Zavrsne Qiskit provere svih sedam registara su prosle.
Log legalne mase: -8.74853634
Log modelske verovatnoce NEXT: -15.55210445
Raspon NEXT kombinacije: 27
Udeo istorijskih raspona manjih ili jednakih NEXT rasponu: 30.010%

CSV: loto7_4716_k89_loto_plus_1736.csv
Redova: 1735
SHA256: 56225ed0f0be7e2d3e30779804d8fa45e96ecfb9d14441eb2a731dd7fff38ae6
Trening prelaza: 1213
Validacionih kola: 260
Test kola: 261

Kandidat: baza 6 red 40 koeficijenata 4 regularizacija 0.001
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 1213
Registar: 1 gubitak: 0.96791923 norma gradijenta: 4.7479463e-08
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97457942 norma gradijenta: 7.8301366e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97488143 norma gradijenta: 1.4781582e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97476524 norma gradijenta: 6.8416974e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97480664 norma gradijenta: 1.3653205e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97439642 norma gradijenta: 2.9981303e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97172711 norma gradijenta: 7.6963371e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 16.73884569
Srednji broj pogodaka: 1.134615
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [72, 104, 62, 21, 1, 0, 0, 0]

Kandidat: baza 6 red 40 koeficijenata 4 regularizacija 0.01
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 1213
Registar: 1 gubitak: 0.96914658 norma gradijenta: 3.668906e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97464234 norma gradijenta: 6.1591049e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97489932 norma gradijenta: 8.935354e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97480221 norma gradijenta: 4.5266474e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97483596 norma gradijenta: 1.8431608e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97448818 norma gradijenta: 7.8983916e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97225654 norma gradijenta: 4.6362398e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 16.67163409
Srednji broj pogodaka: 1.130769
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [73, 103, 62, 21, 1, 0, 0, 0]

Kandidat: baza 6 red 40 koeficijenata 8 regularizacija 0.001
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 1213
Registar: 1 gubitak: 0.96647841 norma gradijenta: 5.2197212e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97425177 norma gradijenta: 4.3768262e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97444159 norma gradijenta: 4.2913644e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97421519 norma gradijenta: 3.8571463e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97454630 norma gradijenta: 2.4077765e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97419076 norma gradijenta: 5.2010634e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97147954 norma gradijenta: 4.2598727e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 16.81812803
Srednji broj pogodaka: 1.169231
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [73, 95, 68, 23, 1, 0, 0, 0]

Kandidat: baza 6 red 40 koeficijenata 8 regularizacija 0.01
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 1213
Registar: 1 gubitak: 0.96774285 norma gradijenta: 9.7264656e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97436059 norma gradijenta: 6.1321706e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97452442 norma gradijenta: 7.2015119e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97433319 norma gradijenta: 8.3335828e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97461200 norma gradijenta: 9.2035609e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97430632 norma gradijenta: 7.2143351e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97204967 norma gradijenta: 7.594967e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 16.74229631
Srednji broj pogodaka: 1.165385
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [74, 94, 68, 23, 1, 0, 0, 0]

Kandidat: baza 7 red 40 koeficijenata 4 regularizacija 0.001
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 1213
Registar: 1 gubitak: 0.96791923 norma gradijenta: 5.9191606e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97457942 norma gradijenta: 2.6438058e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97488143 norma gradijenta: 2.0341995e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97476524 norma gradijenta: 1.1602151e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97480664 norma gradijenta: 5.4874474e-08
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97439642 norma gradijenta: 3.0592193e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97172711 norma gradijenta: 2.9440919e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 16.73883787
Srednji broj pogodaka: 1.134615
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [72, 104, 62, 21, 1, 0, 0, 0]

Kandidat: baza 7 red 40 koeficijenata 4 regularizacija 0.01
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 1213
Registar: 1 gubitak: 0.96914658 norma gradijenta: 2.1156049e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97464234 norma gradijenta: 9.5384224e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97489932 norma gradijenta: 5.194207e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97480221 norma gradijenta: 1.5579445e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97483596 norma gradijenta: 7.6719393e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97448818 norma gradijenta: 9.04135e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97225654 norma gradijenta: 8.0905195e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 16.67163643
Srednji broj pogodaka: 1.130769
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [73, 103, 62, 21, 1, 0, 0, 0]

Kandidat: baza 7 red 40 koeficijenata 8 regularizacija 0.001
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 1213
Registar: 1 gubitak: 0.96647841 norma gradijenta: 9.1636009e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97425177 norma gradijenta: 1.6091303e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97444159 norma gradijenta: 5.3420094e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97421519 norma gradijenta: 3.055012e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97454630 norma gradijenta: 6.9776943e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97419076 norma gradijenta: 9.596781e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97147954 norma gradijenta: 6.2466205e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 16.81813438
Srednji broj pogodaka: 1.169231
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [73, 95, 68, 23, 1, 0, 0, 0]

Kandidat: baza 7 red 40 koeficijenata 8 regularizacija 0.01
UCENJE ZA VALIDACIJU
Istorijskih prelaza: 1213
Registar: 1 gubitak: 0.96774285 norma gradijenta: 8.4940154e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97436059 norma gradijenta: 5.6271742e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97452442 norma gradijenta: 6.2011174e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97433319 norma gradijenta: 9.3132329e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97461200 norma gradijenta: 7.2898214e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97430632 norma gradijenta: 6.8515242e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97204967 norma gradijenta: 5.7650653e-07
Status optimizatora: Optimization terminated successfully.
VALIDACIONI REZULTAT
Broj kola: 260
Srednji NLL: 16.74229755
Srednji broj pogodaka: 1.165385
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [74, 94, 68, 23, 1, 0, 0, 0]

Izabrana modularna baza: 6
QPE aritmeticki red: 40
Broj koeficijenata: 4
Regularizacija: 0.01
UCENJE ZA ODVOJENI TEST
Istorijskih prelaza: 1473
Registar: 1 gubitak: 0.96922106 norma gradijenta: 5.9945707e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97469379 norma gradijenta: 2.5619317e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97490916 norma gradijenta: 8.9908488e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97478425 norma gradijenta: 9.1504702e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97486029 norma gradijenta: 1.4409547e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97456364 norma gradijenta: 9.6056892e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97246891 norma gradijenta: 7.432634e-07
Status optimizatora: Optimization terminated successfully.
ODVOJENI TEST
Broj kola: 261
Srednji NLL: 16.63961462
Srednji broj pogodaka: 1.272031
Nulte ciljne verovatnoce: 0
Broj kola po pogocima od 0 do 7: [59, 103, 71, 25, 3, 0, 0, 0]
ZAVRSNO UCENJE NA SVIM ISTORIJSKIM PRELAZIMA
Istorijskih prelaza: 1734
Registar: 1 gubitak: 0.96926729 norma gradijenta: 5.5512998e-07
Status optimizatora: Optimization terminated successfully.
Registar: 2 gubitak: 0.97468614 norma gradijenta: 5.5273473e-07
Status optimizatora: Optimization terminated successfully.
Registar: 3 gubitak: 0.97490269 norma gradijenta: 9.3311726e-07
Status optimizatora: Optimization terminated successfully.
Registar: 4 gubitak: 0.97484007 norma gradijenta: 7.631276e-07
Status optimizatora: Optimization terminated successfully.
Registar: 5 gubitak: 0.97486037 norma gradijenta: 8.6176341e-07
Status optimizatora: Optimization terminated successfully.
Registar: 6 gubitak: 0.97460023 norma gradijenta: 9.0669885e-07
Status optimizatora: Optimization terminated successfully.
Registar: 7 gubitak: 0.97227632 norma gradijenta: 6.8106107e-07
Status optimizatora: Optimization terminated successfully.
Zavrsne Qiskit provere svih sedam registara su prosle.
Log legalne mase: -9.08839617
Log modelske verovatnoce NEXT: -15.22231573
Raspon NEXT kombinacije: 25
Udeo istorijskih raspona manjih ili jednakih NEXT rasponu: 19.135%


ZAVRSNE NEXT KOMBINACIJE

loto7_4716_k89.csv
NEXT: 6 x 20 y 29 z 39

loto7_4716_k89_loto_2980.csv
NEXT: 12 x 17 y 26 z 39

loto7_4716_k89_loto_plus_1736.csv
NEXT: 9 x 17 y 28 z 34
"""



"""
Q6-Q10	                            Osnovna ideja	                                                        Primena
Single Qubit Gates	                Rotacije, relativne faze i promene merne baze.	                        Učenje parametara jednokubitnih kapija. 
Multiple Qubits and Entanglement	Tenzorski proizvod, kontrolisane kapije i zapletenost.	                Učenje zapletenih stanja unutar šestokubitnih registara. 
Shor	                            Modularna eksponencijacija, procena faze i nalaženje reda.	            Posebna matematička veza sa istorijom kola. 
Bernstein-Vazirani	                Oporavak skrivenog niza iz linearnog binarnog orakla.	                Orakl izveden iz podataka koji zadovoljava taj uslov. 
Simon	                            Otkrivanje skrivene XOR simetrije funkcije.	                            Funkcija sa odgovarajućom strukturom preslikavanja. 


 6. Quantum_Single_Qubit_Gates.py
 7. Quantum_Entanglement.py
 8. Shor
 9. Bernstein-Vazirani
10. Simon

Za poslednja tri prvo treba definisati orakl iz istorije i proveriti uslove izvornog algoritma. 



--------
6. 

Quantum_Single_Qubit_Gates

Jednokubitna rotacija i fazna interferencija. 
Svaki šestokubitni registar ima proizvoljno stanje. 
Parametri kapija iz istorijskih prelaza kroz regresiju raspodele.



-------------
7. 

Quantum_Entanglement

Kontrolisane kapije i zapletenost. 
Između slojeva naučenih rotacija koriste se kontrolisane operacije. 
Provera redukovanih stanja pokazuje da li kolo zaista stvara zapletenost.

Rotacije zavisne od istorije pripremaju stanje, zatim CNOT lanac povezuje šest kubita, a završne rotacije menjaju mernu bazu. 
Provera zapletenosti preko čistoće redukovanih stanja i poređenje amplituda sa Qiskitom pre obuke.

Koristi naučene rotacije i CNOT lanac. 
Zapletenost se proverava unutar svakog šestokubitnog registra preko redukovanih matrica gustine. 
Redosled bitova prati Qiskit 1.4.

- smer i inverziju CNOT permutacija
- kompleksni gradijent kroz RY, RZ i CNOT
- faktor gradijenta Brierovog gubitka
- Bell i GHZ provere
- čistoća redukovanih stanja i Qiskit redosled bitova
- logaritamska normalizacija, globalni maksimum i hronološka podela 


Svaki kubit koristi niz RY → RZ → RY, koji pretvara relativnu fazu u promenu izlaznih verovatnoća. 
Prozor istorije bira se validacijom. 
Konačna kombinacija dobija se kao globalni maksimum raspodele nad svim legalnim izborima sedam brojeva.

Koristim sedam šestokubitnih registara i niz RY → RZ → RY na svakom kubitu. 
Prozore od jednog i tri prethodna kola biram validacijom. 
Optimizator uči parametre kapija iz cele trening istorije.


- izvodi verovatnoća po sva tri ugla RY → RZ → RY
- prefiksni i sufiksni proizvodi u gradijentu
- dimenzije parametara za oba prozora istorije
- redosled bitova prema Qiskitu
- logaritamska normalizacija i globalni izbor maksimuma
- granice treninga, validacije, testa i završnog učenja.



-------------
8. 

Shorov deo modularna eksponencijacija, QPE, inverzna QFT, procena faze i verizni ralomci za nalaženje reda. 

Orakl je \(f_a(x)=a^x\bmod 41\), uz izbor baze a hronološkom validacijom. 
Modul 41 omogućava modularni inverz za svaki broj od 1 do 39. 
QPE odredjuje red, a dodatni model uči amplitude koraka po modularnoj orbiti. 
Superpozicija periodičnih sopstvenih stanja modularnog operatora, čiju dimenziju određuje red dobijen QPE postupkom. 
Provera Shorove faktorizacije broja 15. 
Nalaženje reda koristiće 12 pomoćnih kubita i šest radnih kubita, a poređenje sa Qiskitom manju QPE konstrukciju.

Prediktivna dopuna uči superpoziciju periodičnih sopstvenih stanja modularnog operatora. 
Koristi modul 41, poslednje kolo kao početak orbite i celu istoriju prelaza za obuku. 
Bazu i broj koeficijenata bira validacijom.

QPE simulira 12 pomoćnih i šest radnih kubita. 
Manja QPE konstrukcija poredi se sa Qiskitom. 
Prediktivni deo koristi sedam šestokubitnih registara.



-------------
9.

Quantum_Bernstein_Vazirani 

Orakl, fazni povratni uticaj i Hadamard transformacije. 
Hadamard transformacije, linearni BV orakl i fazni povratni uticaj. 
Za Loto učim dodatnu fazu orakla iz istorijskih prelaza. 
Prethodno kolo je poznati ulaz. Buduća kombinacija se ne ugrađuje u orakl.

Koristim sedam registara sa po šest kubita i po jedan pomoćni kubit. 
Registri se simuliraju zasebno. 



----------------
10. 

Quantum_Simon

Sačuvaću orakl sa XOR periodom i rešavanje jednačina nad GF(2). 

Amplitude na parovima stanja \(x\) i \(x \oplus b\). 
Kandidati za XOR period biraju se iz treninga, a konačni model hronološkom validacijom. 
Simonovo kolo proverava period konstruisanog orakla. 

Dvostruki registar, orakl sa XOR periodom, Hadamard transformacije i rešavanje jednačina nad GF(2). 
Amplitude na Simonovim parovima stanja. 
Kandidati za period procenjuju se na treningu, a konačni model bira hronološkom validacijom.

Simonova provera koristi 12 kubita. 
Predikcija koristi sedam zasebnih registara sa po šest kubita. 
"""
