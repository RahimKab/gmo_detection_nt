VENV_PYTHON = "/home/strange/Documents/master_2/internship/model/.venv/bin/python"


rule all:
	input:
		"data/raw/.collect_done",
		"data/processed/.cleaning_done",
		"result/.train_done",
		"result/.save_model_done",
		"result/best_model.pt",
		"result/history.json",
		"result/merged_model/classifier.pt",
		"result/merged_model/config.pt"


rule collect:
	output:
		done="data/raw/.collect_done"
	log:
		"result/snakemake_collect.log"
	shell:
		"""
		mkdir -p data/raw result
		{VENV_PYTHON} scripts/prepatation/collect.py > {log} 2>&1
		touch {output.done}
		"""


rule cleaning:
	input:
		"data/raw/.collect_done"
	output:
		done="data/processed/.cleaning_done",
		dataset="data/processed/dataset.csv",
		train_split="data/processed/splits/train.csv",
		val_split="data/processed/splits/val.csv",
		test_split="data/processed/splits/test.csv"
	log:
		"result/snakemake_cleaning.log"
	shell:
		"""
		mkdir -p data/processed/splits result
		{VENV_PYTHON} scripts/prepatation/cleaning.py > {log} 2>&1
		touch {output.done}
		"""


rule train:
	input:
		"data/processed/.cleaning_done"
	output:
		done="result/.train_done",
		model="result/best_model.pt",
		history="result/history.json"
	log:
		"result/snakemake_train.log"
	shell:
		"""
		mkdir -p result
		{VENV_PYTHON} scripts/train.py > {log} 2>&1
		touch {output.done}
		"""


rule save_model:
	input:
		"result/.train_done",
		"result/best_model.pt"
	output:
		done="result/.save_model_done",
		classifier="result/merged_model/classifier.pt",
		config="result/merged_model/config.pt"
	log:
		"result/snakemake_save_model.log"
	shell:
		"""
		mkdir -p result
		{VENV_PYTHON} scripts/utils/save_model.py > {log} 2>&1
		touch {output.done}
		"""


rule clean_collect:
	shell:
		"""
		rm -f data/raw/.collect_done
		"""


rule clean_cleaning:
	shell:
		"""
		rm -f data/processed/.cleaning_done
		rm -f data/processed/dataset.csv
		rm -f data/processed/splits/train.csv
		rm -f data/processed/splits/val.csv
		rm -f data/processed/splits/test.csv
		"""


rule clean_train:
	shell:
		"""
		rm -f result/.train_done
		rm -f result/best_model.pt
		rm -f result/checkpoint.pt
		rm -f result/history.json
		rm -f result/logs.txt
		rm -rf result/plots
		"""


rule clean_save_model:
	shell:
		"""
		rm -f result/.save_model_done
		rm -f result/snakemake_save_model.log
		rm -rf result/merged_model
		"""
