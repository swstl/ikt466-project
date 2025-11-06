REMOTE = main.intro-machine-learning.sanderw.coder:ikt466-project

sync:
	tar czf - --exclude='*.pyc' --exclude='__pycache__' src/ *.py 2>/dev/null | \
		ssh main.intro-machine-learning.sanderw.coder "cd ikt466-project && tar xzf -"
	@echo "Sync complete!"

.PHONY: sync
