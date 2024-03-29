# contains the following attributes:
# ksm_py_path_str
# ksm_top_path_str
# Lith_order
# lith_name_map
# intrusions
# k_intrusions
# p_intrusions
# breccias
# wallrock
# other
# unid
# compiled_liths
# t_fill
# titles

from pathlib import Path

ksm_py_path_str = (
    Path().home()
    / "Life Cycle Geo, LLC/LCG Server - Documents/General/PROJECTS/2019/05 19KSM01 - Seabridge Gold - KSM Project/03 Technical Work/00 Python_R"
)

# JVG Alternative path
ksm_py_path_str_alt = (
    Path().home()
    / "Life Cycle Geo, LLC/LCG Server - General/PROJECTS/2019/05 19KSM01 - Seabridge Gold - KSM Project/03 Technical Work/00 Python_R"
)

ksm_top_path_str = (
    Path().home()
    / "Life Cycle Geo, LLC/LCG Server - Documents/General/PROJECTS/2019/05 19KSM01 - Seabridge Gold - KSM Project/"
)

# JVG Alternative path
ksm_top_path_str_alt = (
    Path().home()
    / "Life Cycle Geo, LLC/LCG Server - General/PROJECTS/2019/05 19KSM01 - Seabridge Gold - KSM Project/"
)


# I really need to consolodqte this with the other one
# actually this just needs its own like "setupksm" file or something
# like not even a package perse, just an environment setup, problem for later
Lith_order = [
    "K1",  # intrusions
    "K2",
    "K3",
    "K4",
    "K5",
    "P1",
    "P1B",
    "P2",
    "P3",
    "P5",
    "P7",
    "P8",  # intrusions
    "PHBX",  # breccias
    "QABX",
    "BXTO",
    "BX",
    "H1",  # wall rock
    "H1B",
    "CGL",
    "SS",
    "SARG",
    "SSC",
    "SSL",
    "SSM",
    "SSN",
    "VABX",
    "VALT",
    "VATF",
    "UNCL",  # unidentified
    "UNID",
    "IU",
    "FLTZ",  # additional codes
    "FLTH",
    "MYLO",
    "BSZ",
    "MTF",
    "SNLS",
    "NREC",
    "FG",  # unique to Sulpherets, see email from Ross Hammett
    "STF",
    "STF2",
    "WR",  # in Alice's list
    "No Data",
]

# replacement names
lith_name_map = {
    "SKA": "BSZ",
    "CVN": "FLTZ",
    "QSBX": "QABX",
    "P4": "P2",
    "QVN": "FLTH",
    "SVN": "MYLO",
    "DDRT": "K4",
    "PMFP": "K3",
    # "PPFP": "P3",  # ask, multiple possible
    "PMON": "P8",
    # "VU": "VABX",  # ask, multiple possible
    "PSYN": "P8",
    "QTVN": "P1",
    "VCGL": "CGL",
}
intrusions = Lith_order[:12]
k_intrusions = Lith_order[:5]
p_intrusions = Lith_order[5:12]
breccias = Lith_order[12:16]
wallrock = Lith_order[16:28]
other = Lith_order[-12:-1]
unid = ["UNCL", "UNID", "IU", "No Data"]

compiled_liths = [
    intrusions,
    k_intrusions,
    p_intrusions,
    breccias,
    wallrock,
    other,
    unid,
]
t_fill = [
    "Intrusions",
    "K-intrusions",
    "P-intrusions",
    "Breccias",
    "Wallrock",
    "Other",
    "Unidentified",
]
titles = [f"Sulphurets PCA bi-plot: {a}" for a in t_fill]
