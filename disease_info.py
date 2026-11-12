"""General, non-prescriptive information shown next to a prediction.
Keys match the class names used in training."""

INFO = {
    "Healthy Leaves": {
        "display": "Healthy leaf",
        "severity": "None detected",
        "about": "No visible sign of the three diseases this model knows (powdery mildew, "
                 "downy mildew, bacterial rot / leaf spot).",
        "symptoms": ["Uniform green colour", "No powdery coating, oily yellow patches or dark spots"],
        "management": ["Keep scouting regularly, especially after humid or rainy spells.",
                       "Maintain an open canopy for airflow and light.",
                       "A healthy prediction on one leaf does not clear the whole vine."],
    },
    "Powdery Mildew": {
        "display": "Powdery mildew",
        "severity": "Can spread quickly on leaves, shoots and berries",
        "about": "A fungal disease (Erysiphe necator). Unlike many other fungi it does not need "
                 "free water on the leaf and is favoured by moderate temperatures and shaded, "
                 "crowded canopies.",
        "symptoms": ["White to grey powdery growth on the leaf surface",
                     "Patches that may merge; leaves can curl or distort",
                     "Berries can be affected and may crack"],
        "management": ["Improve airflow and sunlight penetration (shoot positioning, leaf removal).",
                       "Remove and dispose of badly infected material.",
                       "Protective programmes work best before symptoms appear; ask your local "
                       "extension service which products are registered and when to apply them."],
    },
    "Downey Mildew": {
        "display": "Downy mildew",
        "severity": "Can be severe in warm, wet weather",
        "about": "A water-mould disease (Plasmopara viticola) favoured by warm, humid or rainy "
                 "conditions and long periods of leaf wetness.",
        "symptoms": ["Pale yellow 'oil-spot' patches on the upper leaf surface",
                     "White downy growth on the underside of the patch in humid weather",
                     "Spots turn brown and leaves may drop early"],
        "management": ["Open the canopy and avoid prolonged leaf wetness (drainage, no overhead watering).",
                       "Remove infected leaves; clean up fallen leaves at season end.",
                       "Follow local disease-forecast advice for protectant timing; confirm "
                       "registered products with your extension service."],
    },
    "Bacterial Rot": {
        "display": "Bacterial rot / leaf spot",
        "severity": "Can reduce yield; spreads with rain and contaminated tools",
        "about": "A bacterial disease (the dataset images are labelled 'bacterial leaf spot', "
                 "commonly Xanthomonas campestris pv. viticola). It spreads by rain splash, "
                 "tools and infected planting material.",
        "symptoms": ["Small dark, angular spots, often with a yellow halo",
                     "Spots may merge into larger dead areas; leaves can tear or drop",
                     "Stems and bunches can also be affected"],
        "management": ["Remove and destroy infected leaves and prunings; disinfect tools.",
                       "Avoid working in the vineyard when plants are wet.",
                       "Bactericide use differs by region - get a diagnosis confirmed locally "
                       "before treating."],
    },
}

DISCLAIMER = ("This tool is a research prototype trained on one public dataset of single leaves "
              "photographed on plain backgrounds. It is not a diagnosis. Confirm with a plant "
              "pathologist or your local agricultural extension service before treating.")