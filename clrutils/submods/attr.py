# lithological order
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
    "K-intrustions",
    "P-intrustions",
    "Breccias",
    "Wallrock",
    "Other",
    "Unidentified",
]
titles = [f"Sulphurets PCA bi-plot: {a}" for a in t_fill]
