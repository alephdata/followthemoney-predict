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

SCHEMAS = set(("Person", "Company", "LegalEntity", "Organization", "PublicBody"))

FIELDS_BAN_SET = set(["alephUrl", "modifiedAt", "retrievedAt", "sourceUrl"])

FEATURE_FTM_COMPARE_WEIGHTS = {
    "name": 0.6,
    "country": 0.1,
    "email": 0.2,
    "phone": 0.3,
    "website": 0.1,
    "incorporationDate": 0.2,
    "dissolutionDate": 0.2,
    "registrationNumber": 0.3,
    "idNumber": 0.3,
    "taxNumber": 0.3,
    "vatCode": 0.3,
    "jurisdiction": 0.1,
    "mainCountry": 0.1,
    "bvdId": 0.3,
    "okpoCode": 0.3,
    "innCode": 0.3,
    "country": 0.1,
    "wikipediaUrl": 0.1,
    "wikidataId": 0.3,
    "address": 0.3,
    "accountNumber": 0.3,
    "iban": 0.3,
    "irsCode": 0.3,
    "cikCode": 0.3,
    "kppCode": 0.3,
    "fnsCode": 0.3,
    "ogrnCode": 0.3,
    "birthDate": 0.2,
    "deathDate": 0.2,
    "nationality": 0.1,
    "passportNumber": 0.3,
}
