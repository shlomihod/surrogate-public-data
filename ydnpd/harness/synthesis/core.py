import warnings

import pandas as pd

import shutil

SYNTHESIZERS = [
    "id",
    "CTGAN",
    "CopulaGAN",
    "GaussianCopula",
    "TVAE",
    "aim",
    "aim_torch",
    "aim_jax",
    "mwem",
    "gem",
    "mst",
    "pacsynth",
    "dpctgan",
    "patectgan",
    "dpgan",
    "privbayes",
    "adsgan",
    "decaf",
    "pategan",
    "AIM",
]


def generate_synthetic_data(
    dataset: pd.DataFrame, schema: dict, epsilon: float, synth_name: str, **hparams
):

    num_samples = len(dataset)

    if synth_name == "id":
        synth_df = dataset.copy()

    # SDV
    elif synth_name in ["CTGAN", "CopulaGAN", "GaussianCopula", "TVAE"]:

        try:
            from sdv import single_table as sdv_single_table
            from sdv.metadata import SingleTableMetadata
        except ImportError:
            warnings.warn("SDV is not installed. Please install it to use SDV synthesizers.")

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(dataset)

        synthesizer = getattr(sdv_single_table, f"{synth_name}Synthesizer")(
            metadata, **hparams
        )
        synthesizer.fit(dataset)
        synth_df = synthesizer.sample(num_samples)

    # SmartNoise
    elif synth_name in ["aim", "mwem", "mst", "pacsynth", "dpctgan", "patectgan"]:

        try:
            from snsynth import Synthesizer as snsynth_Synthesizer
        except ImportError:
            warnings.warn(
                "SmartNoise is not installed. Please install it to use SmartNoise synthesizers."
            )

        continuous_columns = []
        categorical_columns = []

        for col, info in schema.items():
            match info["dtype"]:
                case "object":
                    categorical_columns.append(col)
                case "int32" | "int64":
                    if len(dataset[col].unique()) > 100:
                        continuous_columns.append(col)
                    else:
                        categorical_columns.append(col)
                case "float64":
                    continuous_columns.append(col)
                case _:
                    print(f"Unknown type: {info['dtype']}")

        preprocessor_eps = hparams.pop("preprocessor_eps")
        synthesizer = snsynth_Synthesizer.create(
            synth_name, epsilon=epsilon, verbose=True, **hparams
        )
        synthesizer.fit(
            dataset,
            preprocessor_eps=preprocessor_eps,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
        )

        synth_df = synthesizer.sample(num_samples)

    # Synthcity
    elif synth_name in ["dpgan", "adsgan", "decaf", "pategan", "AIM"]:

        try:
            from synthcity.plugins import Plugins
            from synthcity.plugins.core.dataloader import GenericDataLoader
        except ImportError:
            warnings.warn(
                "SynthCity is not installed. Please install it to use SynthCity synthesizers."
            )

        loader = GenericDataLoader(synth_name)

        synthesizer = Plugins().get(synth_name.lower(), epsilon=epsilon, **hparams)

        synthesizer.fit(loader)

        synth_df = synthesizer.generate(count=num_samples).dataframe()

    elif synth_name == "privbayes":
        from ydnpd.harness.synthesis.privbayes import PrivBayes

        synthesizer = PrivBayes(epsilon=epsilon, **hparams)
        synth_df = synthesizer.fit_sample(dataset, schema)

    elif synth_name == "aim_jax":
        from ydnpd.harness.synthesis.aim_jax import AIMSynthesizerJax

        continuous_columns = []
        categorical_columns = []

        for col, info in schema.items():
            match info["dtype"]:
                case "object":
                    categorical_columns.append(col)
                case "int32" | "int64":
                    if len(dataset[col].unique()) > 100:
                        continuous_columns.append(col)
                    else:
                        categorical_columns.append(col)
                case "float64":
                    continuous_columns.append(col)
                case _:
                    print(f"Unknown type: {info['dtype']}")

        preprocessor_eps = hparams.pop("preprocessor_eps")
        synthesizer = AIMSynthesizerJax(epsilon=epsilon, verbose=True, **hparams)
        synthesizer.fit(
            dataset,
            preprocessor_eps=preprocessor_eps,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
        )
        synth_df = synthesizer.sample(num_samples)

    
    elif synth_name == "gem":
        from ydnpd.harness.synthesis.gem import GEMSynthesizer

        continuous_columns = []
        categorical_columns = []

        for col, info in schema.items():
            match info["dtype"]:
                case "object":
                    categorical_columns.append(col)
                case "int32" | "int64":
                    if len(dataset[col].unique()) > 100:
                        continuous_columns.append(col)
                    else:
                        categorical_columns.append(col)
                case "float64":
                    continuous_columns.append(col)
                case _:
                    print(f"Unknown type: {info['dtype']}")

        preprocessor_eps = hparams.pop("preprocessor_eps")

        synthesizer = GEMSynthesizer(epsilon=epsilon, verbose=True, **hparams)
        synthesizer.fit(
            dataset,
            preprocessor_eps=preprocessor_eps,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
        )
        synth_df = synthesizer.sample(num_samples)

        # NOTE: one fix, since the synthesizer does not return the same named columns
        synth_df.columns = dataset.columns

        # NOTE: two fix, make sure the columns are right domain
        for col in categorical_columns:
            old_values = schema[col]["values"] 
            sorted_keys = sorted(old_values.keys())
            
            decode_map_int = { i: original_code for i, original_code in enumerate(sorted_keys) }
            
            synth_df[col] = synth_df[col].map(decode_map_int)

        # let's clear the cache, which is a folder under synthesizer.algo.default_dir
        shutil.rmtree(synthesizer.algo.default_dir, ignore_errors=True)

    else:
        raise ValueError(f"Unknown synthesizer name: {synth_name}")

    return synth_df
