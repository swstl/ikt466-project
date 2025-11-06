include .env

sync:
	tar czf - --exclude='*.pyc' --exclude='__pycache__' src/ *.py 2>/dev/null | \
		ssh $(SSH_HOST) "cd $(REMOTE_DIR) && tar xzf -"
	@echo "Sync complete!"

.PHONY: sync
