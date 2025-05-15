class PromptTemplate:
    def __init__(self, input_variables, template):
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs):
        return self.template.format(**kwargs)


prompt_template = open("task/manus准确性任务prompt.txt", "r").read()
prompt = PromptTemplate(
    input_variables=["query", "session_text"], template=prompt_template
)
print(prompt.format(query="你好吗", session_text="你好，我很好，谢谢！"))
# 这里是一个简单的测试代码，使用了PromptTemplate类来格式化一个prompt
