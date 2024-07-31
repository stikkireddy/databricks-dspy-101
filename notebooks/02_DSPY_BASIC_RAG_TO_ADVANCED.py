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
    model_id="dspy102",  # name this to whatever you would like
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

colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

dspy.settings.configure(lm=lm, rm=colbertv2_wiki17_abstracts)

# COMMAND ----------

from dspy.datasets import HotPotQA

# Load the dataset.
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)

# COMMAND ----------

# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]

len(trainset), len(devset)

# COMMAND ----------


class QuestionAnswerContextSignature(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(QuestionAnswerContextSignature)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)
    
rag = RAG()

# COMMAND ----------

rag("What is machine learning?")

# COMMAND ----------

from dspy.teleprompt import BootstrapFewShot

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 4-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=4, max_labeled_demos=4, teacher_settings={"lm": teacher_lm, "rm": colbertv2_wiki17_abstracts})

# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(metric=validate_context_and_answer, **config)

# Compile!
compiled_rag = teleprompter.compile(RAG(), trainset=trainset)

# COMMAND ----------

# Ask any question you like to this simple RAG program.
my_question = "What castle did David Gregory inherit?"

# Get the prediction. This contains `pred.context` and `pred.answer`.
compiled_rag(my_question)

# COMMAND ----------

from dspy.evaluate import Evaluate

# Set up the evaluator, which can be used multiple times.
evaluate = Evaluate(devset=gsm8k_devset, metric=gsm8k_metric, num_threads=4, display_progress=True, display_table=True)

# Evaluate our `optimized_cot` program.
evaluate(optimized_cot)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Simplified Baleen Architecture 
# MAGIC
# MAGIC 1. generate search queries
# MAGIC 2. retrieve
# MAGIC 3. cot answer
# MAGIC 4. repeat

# COMMAND ----------

class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

from dsp.utils import deduplicate

class SimplifiedBaleen(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.answer)

# COMMAND ----------

# Ask any question you like to this simple RAG program.
my_question = "How many storeys are in the castle that David Gregory inherited?"

# Get the prediction. This contains `pred.context` and `pred.answer`.
uncompiled_baleen = SimplifiedBaleen()  # uncompiled (i.e., zero-shot) program
uncompiled_baleen(my_question)

# COMMAND ----------

from dspy.teleprompt import BootstrapFewShot

def validate_context_and_answer_and_hops(example, pred, trace=None):
    if not dspy.evaluate.answer_exact_match(example, pred): return False
    if not dspy.evaluate.answer_passage_match(example, pred): return False

    hops = [example.question] + [outputs.query for *_, outputs in trace if 'query' in outputs]

    if max([len(h) for h in hops]) > 100: return False
    if any(dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8) for idx in range(2, len(hops))): return False

    return True
  


config = dict(teacher_settings={"lm": teacher_lm, "rm": colbertv2_wiki17_abstracts})

optimizer = BootstrapFewShot(metric=validate_context_and_answer_and_hops, **config)
compiled_baleen = optimizer.compile(SimplifiedBaleen(), teacher=SimplifiedBaleen(passages_per_hop=2), trainset=trainset)

# COMMAND ----------

my_question = "How many storeys are in the castle that David Gregory inherited?"

compiled_baleen(my_question)

# COMMAND ----------

compiled_baleen.save("./compiled_json.json")

# COMMAND ----------



# COMMAND ----------

# MAGIC %environment
# MAGIC "client": "1"
# MAGIC "base_environment": ""
