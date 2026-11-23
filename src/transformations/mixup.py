"""
Mixup pentru volumele RMN din Stanford MRNet.

Volumele pot avea numere diferite de felii.
Aliniem numarul de felii prin interpolare liniara
inainte de a combina cele doua volume.
"""

import torch


def aliniaza_felii(volum: torch.Tensor, nr_felii: int) -> torch.Tensor:
    """
    Redimensioneaza numarul de felii al unui volum RMN.

    Intrare:
        volum: [S, C, H, W]
        nr_felii: numarul dorit de felii

    Iesire:
        [nr_felii, C, H, W]
    """

    if volum.ndim != 4:
        raise ValueError(
            f"Se asteapta [S, C, H, W], primit {tuple(volum.shape)}"
        )

    if volum.shape[0] < 1 or nr_felii < 1:
        raise ValueError("Numarul de felii trebuie sa fie pozitiv.")

    if volum.shape[0] == nr_felii:
        return volum

    # Pozitiile noilor felii pe axa volumului
    pozitii = torch.linspace(
        0,
        volum.shape[0] - 1,
        nr_felii,
        device=volum.device
    )

    # Cele doua felii vecine pentru fiecare pozitie
    jos = pozitii.floor().long()
    sus = pozitii.ceil().long()

    # Coeficientul de interpolare
    alpha = (
        (pozitii - jos)
        .to(volum.dtype)
        .view(-1, 1, 1, 1)
    )

    # Interpolare liniara intre felii
    return (
        volum[jos] * (1 - alpha)
        + volum[sus] * alpha
    )


def mixup_volume(
    volum_a: torch.Tensor,
    volum_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """
    Combina doua volume RMN folosind Mixup.

    Formula:
        volum_mixat = lam * A + (1 - lam) * B

    Pastreaza dimensiunile volumului A.
    """

    if not (0.0 <= lam <= 1.0):
        raise ValueError("Lambda trebuie sa fie intre 0 si 1.")

    if (
        volum_a.ndim != 4
        or volum_b.ndim != 4
        or volum_a.shape[1:] != volum_b.shape[1:]
    ):
        raise ValueError(
            f"Volume incompatibile: "
            f"{tuple(volum_a.shape)}, {tuple(volum_b.shape)}"
        )

    # Aliniem numarul de felii al volumului B
    volum_b = aliniaza_felii(
        volum_b,
        volum_a.shape[0]
    )

    # Aplicam Mixup
    volum_mixat = (
        lam * volum_a
        + (1.0 - lam) * volum_b
    )

    return volum_mixat