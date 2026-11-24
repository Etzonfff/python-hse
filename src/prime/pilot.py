import math
from statistics import NormalDist

z = NormalDist().inv_cdf

def n_per_group(sigma: float, mde: float, alpha: float = 0.05, power: float = 0.8,
                comparisons: int = 1) -> int:
    """Клиентов на КАЖДУЮ группу, чтобы заметить эффект mde с заданной мощностью."""
    za = z(1 - alpha / (2 * comparisons))
    # Второй квантиль отвечает за мощность. Какой у него аргумент?
    zb = z(power)
    return math.ceil(2 * (za + zb) ** 2 * sigma ** 2 / mde ** 2)


def mde_for_n(
    sigma: float,
    n: int,
    alpha: float = 0.05,
    power: float = 0.8,
    comparisons: int = 1,
) -> float:
    """Какой эффект обнаруживаем с заданной мощностью при n клиентах на группу."""
    za = z(1 - alpha / (2 * comparisons))
    zb = z(power)
    return (za + zb) * sigma * math.sqrt(2 / n)


def diff_ci(t, c, alpha: float = 0.05) -> dict:
    """Разность test − control, доверительный интервал и p-значение."""
    d = t.mean() - c.mean()
    se = (t.var() / len(t) + c.var() / len(c)) ** 0.5

    q = NormalDist().inv_cdf(1 - alpha / 2)
    p = 2 * (1 - NormalDist().cdf(abs(d / se)))

    return {
        "diff": d,
        "lo": d - q * se,
        "hi": d + q * se,
        "p": p,
    }


def srm_pvalue(sizes) -> float:
    """Проверка равных долей для двух или трёх групп."""
    n = sum(sizes)
    exp = n / len(sizes)
    chi2 = sum((s - exp) ** 2 / exp for s in sizes)

    if len(sizes) == 2:
        return math.erfc(math.sqrt(chi2 / 2))

    if len(sizes) == 3:
        return math.exp(-chi2 / 2)

    raise ValueError(
        "Для 4 и более групп используйте scipy.stats.chisquare"
    )


def cuped(y, x):
    """Возвращает скорректированную метрику и коэффициент theta."""
    theta = y.cov(x) / x.var()
    return y - theta * (x - x.mean()), theta