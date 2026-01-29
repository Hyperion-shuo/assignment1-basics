import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# 编译正则
pattern = re.compile(PAT)

def tokenize(text):
    # findall 会返回所有匹配的列表
    return pattern.findall(text)

# --- 测试案例 ---

# 1. 英语缩写测试 (Contractions)
text1 = "It's don't I'll we're"
print(f"原文: '{text1}'")
print(f"分词: {tokenize(text1)}\n")

# 2. 普通句子与空格处理 (Spaces attached to words)
text2 = "Hello world, this is a test."
print(f"原文: '{text2}'")
print(f"分词: {tokenize(text2)}\n")

# 3. 数字与符号 (Numbers and Symbols)
text3 = "Price: $100.50  (20% off)"
print(f"原文: '{text3}'")
print(f"分词: {tokenize(text3)}\n")

# 4. 多语言支持 (Unicode)
text4 = "Hello你好123测试"
print(f"原文: '{text4}'")
print(f"分词: {tokenize(text4)}\n")

# 5. 连续空格与结尾空格
text5 = "   Start   end   "
print(f"原文: '{text5}'")
print(f"分词: {tokenize(text5)}\n")

# 6. find iter示例
text = "It's 100"

print(f"{'Token':<10} | {'Start':<5} | {'End':<5}")
print("-" * 25)

for match in pattern.finditer(text):
    token = match.group()
    start = match.start()
    end = match.end()
    print(f"{repr(token):<10} | {start:<5} | {end:<5}")
    
# 7. 捕获组
text = "It's"
# 1. 使用普通括号 (Capturing)
# 正则引擎认为：你加了括号，说明你特别想要括号里的内容
print(re.findall(r"'(s|t)", text))
# 输出: ['s'] 
# (注意：它把单引号丢了，只返回了括号里的 s)

# 2. 使用非捕获组 (Non-capturing)
# 正则引擎认为：括号只是为了逻辑，我要返回整体匹配到的东西
print(re.findall(r"'(?:s|t)", text))
# 输出: ["'s"] 
# (注意：这是我们想要的，完整的 token)