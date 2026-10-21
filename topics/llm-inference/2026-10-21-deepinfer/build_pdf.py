import re, sys, os, pathlib
R = pathlib.Path(os.environ.get("REPORTS_DIR", "reports"))  # 原稿目录
B = R/"deliverables/build"
files = [R/"01_产品对比_DeepInfer与推理引擎竞品.md", R/"02_推理Infra技术调研与解析.md", R/"03_同口径基准测试方案.md",
         R/"topics/00_专题索引与勘误.md"] + sorted((R/"topics").glob("T[1-9]_*.md"))
rep = {"✅":"[是]","🟡":"[部分]","❌":"[否]","❓":"[待核]","⚠️":"[!]","⚠":"[!]","🤖":"","→":"→","✓":"[√]","✗":"[×]","★":"*","🚀":"","►":">","◄":"<","⁹":"^9","⁸":"^8","⁶":"^6","³":"^3","²":"^2","¹":"^1","⁰":"^0","⁻":"^-"}
out = []
for i,f in enumerate(files):
    t = f.read_text(encoding="utf-8")
    for k,v in rep.items(): t = t.replace(k,v)
    t = re.sub(r"[\U0001F300-\U0001FAFF☀-⛿✀-➿️]", "", t)
    lines, res, infence = t.split("\n"), [], False
    li = re.compile(r"^\s*([-*+]|\d+\.)\s+")
    for ln in lines:
        if ln.lstrip().startswith("```"): infence = not infence
        if (not infence and li.match(ln) and res and res[-1].strip()
                and not li.match(res[-1]) and not res[-1].lstrip().startswith(("|",">")) and not res[-1].startswith(" ")):
            res.append("")
        res.append(ln)
    t = "\n".join(res)
    # demote: each doc's H1 stays H1 (new chapter)
    out.append(f"\n\n\\newpage\n\n" + t if i else t)
(B/"all.md").write_text("\n".join(out), encoding="utf-8")
print(len(files), "files merged")
