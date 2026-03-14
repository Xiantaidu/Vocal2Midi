import pathlib

import click
import pandas as pd
import yaml


@click.command()
@click.option("--data_folder",
              type=click.Path(exists=True, file_okay=False),
              required=True,
              help="Root directory containing dataset folders")
@click.option("--language",
              default="zh",
              show_default=True,
              help="Language identifier for the dataset")
@click.option("--abs", "absolute",
              is_flag=True,
              help="Use absolute paths (default: relative)")
@click.option("--output",
              default=None,
              help="Output config filename (auto-generated if not specified)")
def main(data_folder, language, absolute, output):
    """Generate DiffSinger dataset config from directory structure"""

    # Handle default output filename
    out_data_config = output or f"datasets_config_{language}.yaml"

    csv_paths = []
    for csv in pathlib.Path(data_folder).rglob("transcriptions.csv"):
        parent_dir = csv.parent

        if not (csv.exists() and (parent_dir / "wavs").exists()):
            print(f"wavs folder does not exist: {parent_dir / 'wavs'}")
            continue

        try:
            df = pd.read_csv(csv, encoding="utf-8")
            if len(df) < 2:
                raise ValueError(f"{csv} must contain at least 2 rows")
            if "ph_seq" not in df.columns:
                raise ValueError(f"{csv} is missing 'ph_seq' column")

            # Determine label type
            label_type = "full" if "ph_dur" in df.columns else "weak"

            # Build path representation
            raw_path = str(parent_dir.absolute()) if absolute else str(parent_dir)

            csv_paths.append({
                "raw_data_dir": raw_path,
                "label_type": label_type,
                "language": language,
                "test_prefixes": [str(df["name"][0])]
            })

        except Exception as e:
            click.echo(f"Error processing {csv}: {str(e)}", err=True)
            continue

    try:
        with open(out_data_config, "w", encoding="utf-8") as f:
            yaml.dump({"datasets": csv_paths}, f, allow_unicode=True)
            click.echo(f"Config successfully generated: {out_data_config}")
    except IOError as e:
        click.echo(f"Failed to write config: {str(e)}", err=True)


if __name__ == "__main__":
    main()
