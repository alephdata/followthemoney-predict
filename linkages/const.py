from pandas import BooleanDtype, CategoricalDtype, StringDtype


PHASES = {
    "train": 0.8,
    "test": 0.2,
}


NEGATIVE_COLLECTION_FIDS = set(
    (
        "88513394757c43089cd44f817f16ca05",  # Khadija Project Research Data
        "45602a9bb6c04a179a2657e56ed3a310",  # Mozambique Persons of Interest (2015)
        "zz_occrp_pdi",  # Persona de Interes (2014)
        "ch_seco_sanctions",  # Swiss SECO Sanctions
        "interpol_red_notices",  # INTERPOL Red Notices
        "45602a9bb6c04a179a2657e56ed3a310",
        # "ru_moscow_registration_2014",  # 3.9GB
        "ru_pskov_people_2007",
        "ua_antac_peps",
        "am_voters",
        "hr_gong_peps",
        "hr_gong_companies",
        "mk_dksk",
        "ru_oligarchs",
        "everypolitician",
        "lg_people_companies",
        "rs_prijave",
        "5b5ec30364bb41999f503a050eb17b78",
        "aecf6ecc4ab34955a1b8f7f542b6df62",
        "am_hetq_peps",
        "kg_akipress_peps",
        # "ph_voters",  # 7.5GB
        "gb_coh_disqualified",
    )
)

FEATURE_KEYS = [
    "name",
    "country",
    "registrationNumber",
    "incorporationDate",
    "address",
    "jurisdiction",
    "dissolutionDate",
    "mainCountry",
    "ogrnCode",
    "innCode",
    "kppCode",
    "fnsCode",
    "email",
    "phone",
    "website",
    "idNumber",
    "birthDate",
    "nationality",
    "accountNumber",
    "iban",
    "wikidataId",
    "wikipediaUrl",
    "deathDate",
    "cikCode",
    "irsCode",
    "vatCode",
    "okpoCode",
    "passportNumber",
    "taxNumber",
    "bvdId",
]
FEATURE_IDXS = dict(zip(FEATURE_KEYS, range(len(FEATURE_KEYS))))

SCHEMAS = set(
    ("Person", "Company", "LegalEntity", "Organization", "PublicBody", "BankAccount")
)

FIELDS_BAN_SET = set(["alephUrl", "modifiedAt", "retrievedAt", "sourceUrl"])
DATAFRAME_FIELDS_TYPES = {
    "name": StringDtype(),
    "schema": CategoricalDtype(SCHEMAS),
    "collection_id": StringDtype(),
    "id": StringDtype(),
    "country": StringDtype(),
    "address": StringDtype(),
    "registrationNumber": StringDtype(),
    "alias": StringDtype(),
    "status": StringDtype(),
    "classification": StringDtype(),
    "gender": StringDtype(),
    "firstName": StringDtype(),
    "lastName": StringDtype(),
    "birthPlace": StringDtype(),
    "birthDate": StringDtype(),
    "idNumber": StringDtype(),
    "motherName": StringDtype(),
    "nationality": StringDtype(),
}
DATAFRAME_META = {
    f"{which}_{c}": t
    for which in ("left", "right")
    for c, t in DATAFRAME_FIELDS_TYPES.items()
}
DATAFRAME_META["judgement"] = BooleanDtype()
DATAFRAME_META["source"] = CategoricalDtype(["linkage", "negative", "positive"])
DATAFRAME_META["phase"] = CategoricalDtype(PHASES.keys())
DATAFRAME_META["features"] = object
DATAFRAME_META["schema"] = StringDtype()
