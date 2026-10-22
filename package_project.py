"""Bundle source, selected weights, split files and the translation deliverables."""

import os
import zipfile

from data import ROOT


def main():
    destination = ROOT.parent / "English_Darija_Translation_Project.zip"
    allowed = {"assets", "logs", "output", "results", "screenshots"}
    with zipfile.ZipFile(
        destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for item in ROOT.iterdir():
            if item.is_file() and (
                item.suffix in {".py", ".ps1", ".md", ".ipynb", ".txt", ".toml"}
                or item.name in {".gitignore", ".gitattributes"}
            ):
                archive.write(item, "rnn_darija_translation/" + item.name)
            elif item.is_dir() and item.name in allowed:
                for base, _, filenames in os.walk(item):
                    for filename in filenames:
                        file = ROOT / base / filename
                        if "_candidate" in filename or (
                            filename.endswith(".pt") and "_lr" in filename
                        ):
                            continue
                        archive.write(
                            file,
                            "rnn_darija_translation/"
                            + file.relative_to(ROOT).as_posix(),
                        )
        for source, target in [
            (".build/build_slides.mjs", "tools/build_slides.mjs"),
            (".build/slides-validation.json", "validation/slides-validation.json"),
        ]:
            if (ROOT / source).exists():
                archive.write(ROOT / source, "rnn_darija_translation/" + target)
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert "rnn_darija_translation/results/vanilla.pt" in archive.namelist()
        assert (
            "rnn_darija_translation/output/Darija_Translation_Report.pdf"
            in archive.namelist()
        )
    print(destination, destination.stat().st_size)


if __name__ == "__main__":
    main()
