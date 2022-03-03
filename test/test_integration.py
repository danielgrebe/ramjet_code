import iso_1D as iso
import pytest

TEST_BASENAME = "test/test_data/test"
TEST_BASENAME_RAYLEIGH = "test/test_data/test_rayleigh"

# Initial Conditions
P0 = 100e3  # Pa
T0 = 300  # K
C_P = 1006
GAMMA = 1.4
MASSE_MOLAIRE = 28.97
ANALYTICAL_ANS = (2.84232747, 0.03454652, 0.09036560, 0.38229732)
RAYLEIGH_ANS = (1.1, 0.31309622314622315, 0.31042060261818183, 1.0087420599478942)


def setup_analytique():
    geo = iso.Geometry(TEST_BASENAME)
    init_cond = iso.InitialConditions(P0, T0)
    gas = iso.IdealGas(GAMMA, MASSE_MOLAIRE, C_P)
    return iso.IsoEcoulement(geo, init_cond, gas)


def iso_analytique():
    flow = setup_analytique()
    return (flow.mach(0.12),
            flow.pressure(0.12)/P0,
            flow.rho(0.12)/flow.gas_prop.density(T0, P0),
            flow.temperature(0.12)/T0)


def setup_rayleigh():
    geo = iso.Geometry(TEST_BASENAME_RAYLEIGH)
    init_cond = iso.InitialConditions(P0, T0)
    gas = iso.IdealGas(GAMMA, MASSE_MOLAIRE, C_P)
    return iso.General_1D_Flow(geo, init_cond, gas, basename=TEST_BASENAME_RAYLEIGH, n=10000)


def rayleigh():
    flow = setup_rayleigh()
    return (flow.mach(0.12),
            flow.pressure(0.12)/P0,
            flow.rho(0.12)/flow.gas_prop.density(T0, P0),
            flow.temperature(0.12)/T0)


def setup_iso_numerique():
    geo = iso.Geometry(TEST_BASENAME)
    init_cond = iso.InitialConditions(P0, T0)
    gas = iso.IdealGas(GAMMA, MASSE_MOLAIRE, C_P)
    return iso.General_1D_Flow(geo, init_cond, gas, heat_on=False)


def iso_numerique():
    flow = setup_iso_numerique()
    return (flow.mach(0.12),
            flow.pressure(0.12)/P0,
            flow.rho(0.12)/flow.gas_prop.density(T0, P0),
            flow.temperature(0.12)/T0)


def test_iso_analytique():
    assert iso_analytique() == pytest.approx(ANALYTICAL_ANS)


def test_iso_numerique():
    assert iso_numerique() == pytest.approx(ANALYTICAL_ANS, 1e-2)


def test_rayleigh():
    assert rayleigh() == pytest.approx(RAYLEIGH_ANS, 1e-2)


def main():
    pass


if __name__ == "__main__":
    main()