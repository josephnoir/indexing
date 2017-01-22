DIRS :=  /Users/noir/Development/caf/wah/build

all:
	@for i in $(DIRS); do $(MAKE) -C $$i $@ || exit; done

test:
	@for i in $(DIRS); do $(MAKE) -C $$i $@ || exit; done

install:
	@for i in $(DIRS); do $(MAKE) -C $$i $@ || exit; done

uninstall:
	@for i in $(DIRS); do $(MAKE) -C $$i $@ || exit; done

clean:
	@for i in $(DIRS); do $(MAKE) -C $$i $@; done

distclean:
	rm -rf $(DIRS) Makefile

doc:
	$(MAKE) -C $(firstword $(DIRS)) $@

.PHONY: all test install uninstall clean distclean
