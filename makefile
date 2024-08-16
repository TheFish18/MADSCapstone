# Directories
DATA_DIR = Data
SCRIPTS_DIR = Scripts
VIZ_DIR = $(DATA_DIR)/Viz
SCRIPTS_VIZ_DIR = $(SCRIPTS_DIR)/Viz
OUTPUT_DIR = main

# Files
MAIN_NOTEBOOK = main.ipynb
MAIN_HTML = $(OUTPUT_DIR)/main.html
VIZ_HTML = $(VIZ_DIR)/sp500_fomc_bb_viz.html
VIZ_SCRIPT = $(SCRIPTS_VIZ_DIR)/sp500_fomc_bb_viz.py

# Rule to convert main.ipynb to HTML
$(MAIN_HTML): $(MAIN_NOTEBOOK) $(VIZ_HTML)
	jupyter nbconvert --to html --execute --no-input --embed-images $(MAIN_NOTEBOOK)
	mv main.html $(MAIN_HTML)

# Rule to generate sp500_fomc_bb_viz.html
$(VIZ_HTML): $(VIZ_SCRIPT)
	python -m Scripts.Viz.sp500_fomc_bb_viz

# Phony targets
.PHONY: all clean

# Default target
all: $(MAIN_HTML) $(VIZ_HTML)

# Clean target to remove generated files
clean:
	rm -f $(MAIN_HTML)