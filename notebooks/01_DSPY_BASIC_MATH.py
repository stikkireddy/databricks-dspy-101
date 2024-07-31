# Databricks notebook source
# MAGIC %pip install -U openai 
# MAGIC %pip install dbtunnel[gradio] dspy-ai pydantic
# MAGIC %pip install arize-otel openai openinference-instrumentation-openai opentelemetry-sdk openinference-instrumentation-dspy opentelemetry-exporter-otlp
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from arize_otel import register_otel, Endpoints

    # Setup OTEL via our convenience function
register_otel(
    endpoints=Endpoints.PHOENIX_LOCAL,
    model_id="dspy101",  # name this to whatever you would like
)

# Import the automatic instrumentor from OpenInference
from openinference.instrumentation.dspy import DSPyInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor

# Start the instrumentor for DSPy
DSPyInstrumentor().instrument()
OpenAIInstrumentor().instrument()

# COMMAND ----------

WORKSPACE_URL = f'https://{spark.conf.get("spark.databricks.workspaceUrl")}'
SERVING_ENDPOINTS_URL = f"{WORKSPACE_URL}/serving-endpoints/"
MODEL_ID = "databricks-meta-llama-3-1-70b-instruct"
TEACHER_MODEL_ID = "databricks-meta-llama-3-1-405b-instruct"
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
WORKSPACE_URL, SERVING_ENDPOINTS_URL, MODEL_ID

# COMMAND ----------

import dspy
lm = dspy.OpenAI(
    model=MODEL_ID,
    api_key=TOKEN,
    api_base=SERVING_ENDPOINTS_URL,
)
teacher_lm = dspy.OpenAI(
    model=TEACHER_MODEL_ID,
    api_key=TOKEN,
    api_base=SERVING_ENDPOINTS_URL,
)
dspy.settings.configure(lm=lm)

lm("What is ml?")

# COMMAND ----------

from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
gsm8k = GSM8K()
gsm8k_trainset, gsm8k_devset = gsm8k.train[:10], gsm8k.dev[:10]
gsm8k_trainset, gsm8k_devset

# COMMAND ----------


class QuestionAnswerSignature(dspy.Signature):
    """Answer the question to the best of your abilities, it will be related to math."""
    question = dspy.InputField(description="The math question to be answered?")
    answer = dspy.OutputField(description="The the answer to the question?")

class QuestionAnswerBasic(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)
      
class QuestionAnswerSignature(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(QuestionAnswerSignature)
    
    def forward(self, question):
        return self.prog(question=question)
      


# COMMAND ----------

from dspy.teleprompt import BootstrapFewShot

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4, teacher_settings={"lm": teacher_lm})

# Optimize! Use the `gsm8k_metric` here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShot(metric=gsm8k_metric, **config)
optimized_cot = teleprompter.compile(QuestionAnswerBasic(), trainset=gsm8k_trainset)

# COMMAND ----------

optimized_cot("What is 20 times 30?")

# COMMAND ----------

from dspy.evaluate import Evaluate

# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=True)

# Evaluate our `optimized_cot` program.
evaluate(optimized_cot)

# COMMAND ----------



# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
