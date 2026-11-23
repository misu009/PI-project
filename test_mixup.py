
import torch
import torch.nn.functional as F

from src.transformations.mixup import (
    aliniaza_felii,
    mixup_volume
)


def test_mixup():

    print("TEST 1: Combinarea a doua volume")

    # Primul volum contine doar valori 1
    volum_a = torch.ones(30, 3, 224, 224)

    # Al doilea volum contine doar valori 0
    volum_b = torch.zeros(38, 3, 224, 224)

    lam = 0.7

    volum_mixat = mixup_volume(
        volum_a,
        volum_b,
        lam
    )

    # Formula: 0.7 * 1 + 0.3 * 0 = 0.7
    rezultat_asteptat = torch.full_like(volum_a, 0.7)

    assert volum_mixat.shape == volum_a.shape
    assert torch.allclose(
        volum_mixat,
        rezultat_asteptat,
        atol=1e-6
    )
    assert torch.isfinite(volum_mixat).all()

    print("PASSED - Mixup calculeaza corect valorile")
    print("Forma rezultatului:", volum_mixat.shape)
    print("Valoarea primului pixel:", volum_mixat[0, 0, 0, 0].item())

    print("\nTEST 2: Interpolarea numarului de felii")

    # 3 felii cu valori 0, 10, 20
    volum = torch.stack([
        torch.full((3, 8, 8), float(i * 10))
        for i in range(3)
    ])

    rezultat = aliniaza_felii(volum, 5)

    # Rezultatul asteptat: 0, 5, 10, 15, 20
    valori = rezultat[:, 0, 0, 0]

    assert rezultat.shape == (5, 3, 8, 8)
    assert torch.allclose(
        valori,
        torch.tensor([0., 5., 10., 15., 20.])
    )

    print("PASSED - Interpolarea functioneaza")
    print("Valori:", valori.tolist())

    print("\nTEST 3: Compatibilitate cu Z-score")

    # Dupa Z-score apar si valori negative
    volum_c = torch.full((20, 3, 8, 8), -1.5)
    volum_d = torch.full((25, 3, 8, 8), 2.5)

    rezultat_zscore = mixup_volume(
        volum_c,
        volum_d,
        0.7
    )

    assert torch.isfinite(rezultat_zscore).all()
    assert torch.allclose(
        rezultat_zscore,
        torch.full_like(volum_c, -0.3),
        atol=1e-6
    )

    print("PASSED - Mixup accepta valori negative")
    print("Valoare obtinuta:", rezultat_zscore[0, 0, 0, 0].item())

    print("\nTEST 4: Loss si calculul gradientilor")

    prediction = torch.tensor(
        [0.2],
        requires_grad=True
    )

    label_a = torch.tensor([1.0])
    label_b = torch.tensor([0.0])

    loss_a = F.binary_cross_entropy_with_logits(
        prediction, label_a
    )

    loss_b = F.binary_cross_entropy_with_logits(
        prediction, label_b
    )

    loss = 0.7 * loss_a + 0.3 * loss_b

    loss.backward()

    assert torch.isfinite(loss).all()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()

    print("PASSED - Loss si gradientii sunt valizi")
    print("Loss:", loss.item())

    print("\nTOATE TESTELE MIXUP AU TRECUT!")


if __name__ == "__main__":
    test_mixup()
