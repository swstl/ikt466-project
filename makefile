include .env

sync:
	tar czf - --exclude='*.pyc' --exclude='__pycache__' src/ *.py 2>/dev/null | \
		ssh $(word 1,$(subst :, ,$(REMOTE))) "cd $(word 2,$(subst :, ,$(REMOTE))) && tar xzf -"
	@echo "Sync complete!"

.PHONY: sync
