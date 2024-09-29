.PHONY: hello
hello:
	@echo "Hello Everyone."

.PHONY: help
help:
	@echo "Usage: make [target]"
	@echo "  hello: Print \"Hello Everyone!\"."

.PHONY: env
env:
	set PYTHONPATH=D:\DataScience_For_mySelf\Projects_myself\RagMLOPS\TriModalRAG_System



# DEVELOPMENT
define rag_env
conda create -p rag_env python==3.9 -y
endef



.PHONY: test
test: 
	python tests/integration/main.py

.PHONY: install
install:
	@$(PIP) install -r requirements.txt


.PHONY: clean
clean:
	python -rf __pycache__

