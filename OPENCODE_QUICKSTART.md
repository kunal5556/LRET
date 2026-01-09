# 🚀 OpenCodeAI Quick Start Card

**Branch:** `agent-MD` ✅  
**Status:** Ready for Testing  
**Date:** January 9, 2026

---

## ⚡ 60-Second Setup

```bash
# 1. Install OpenCode (one-time)
curl -fsSL https://opencode.ai/install | bash

# 2. Restart terminal, then navigate to repo
cd /Users/suryanshsingh/Documents/LRET

# 3. Ensure you're on agent-MD branch
git checkout agent-MD

# 4. Run verification script
./test_opencode.sh

# 5. Start OpenCode
opencode
```

---

## 💬 First Test Queries

Once in OpenCode terminal, try these:

```
@agent.md What is LRET?

@agent.md How do I build the project?

@agent.md Show me the quantum error correction code

@agent.md Run test_simple and explain results

@agent.md Create a simple quantum circuit example
```

---

## 📁 Key Files

| File | Purpose | Lines |
|------|---------|-------|
| `agent.md` | Main AI agent configuration | 18,548 |
| `AGENT_GUIDE.md` | User documentation | 509 |
| `OPENCODE_TESTING_GUIDE.md` | Full testing guide | NEW |
| `test_opencode.sh` | Verification script | NEW |

---

## 🔧 If Something Breaks

```bash
# Rebuild from scratch
cd build
rm -rf *
cmake ..
make -j$(sysctl -n hw.ncpu)

# Resync with feature branch (if needed)
git checkout agent-MD
git rebase origin/feature/framework-integration
git push origin agent-MD --force-with-lease

# Verify agent file
head -100 agent.md
wc -l agent.md  # Should be ~18,548 lines
```

---

## 📊 Success Checklist

- [ ] OpenCode installed and running
- [ ] Agent responds to `@agent.md` queries
- [ ] Build commands work via agent
- [ ] Test binaries execute via agent
- [ ] Python integration works
- [ ] No crashes or errors

**Target: 85/100 score** (see full testing guide)

---

## 📞 Resources

- **Full Guide:** [OPENCODE_TESTING_GUIDE.md](OPENCODE_TESTING_GUIDE.md)
- **Agent Config:** [agent.md](agent.md)
- **User Guide:** [AGENT_GUIDE.md](AGENT_GUIDE.md)
- **OpenCode Docs:** https://docs.opencode.ai

---

## 🎯 What We're Testing

1. **Recognition:** Does OpenCode load agent.md correctly?
2. **Navigation:** Can it find files and understand structure?
3. **Execution:** Can it build, test, and run code?
4. **Integration:** Does Python/Docker workflow work?
5. **Robustness:** How does it handle errors?

---

**Pro Tip:** Start with simple queries, then progress to complex ones. Document issues in the testing guide's tracking template.
