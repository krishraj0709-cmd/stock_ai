# training/write_scanner.py
code = open("dashboard/scanner.py", "r", encoding="utf-8").read()

# Fix the broken line
code = code.replace("st.rerun()s", "st.rerun()")
code = code.replace("st.rerun()S", "st.rerun()")

with open("dashboard/scanner.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Fixed scanner.py")