from transformer_trainer import translate

text = "i will not come tomorrow"
output = translate(text)

print("\n📤 Translation Result")
print(f"Input:  {text}")
print(f"Output: {output}")
