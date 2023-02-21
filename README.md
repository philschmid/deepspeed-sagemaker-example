# Fine-tune FLAN-T5 XL/XXL using DeepSpeed on Amazon SageMaker

FLAN-T5, released with the [Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf) paper, is an enhanced version of T5 that has been fine-tuned in a mixture of tasks, or simple words, a better T5 model in any aspect. FLAN-T5 outperforms T5 by double-digit improvements for the same number of parameters. Google has open sourced [5 checkpoints available on Hugging Face](https://huggingface.co/models?other=arxiv:2210.11416) ranging from 80M parameter up to 11B parameter.

In a previous blog post, we already learned how to [“Fine-tune FLAN-T5 XL/XXL using DeepSpeed & Hugging Face Transformers”](https://www.philschmid.de/fine-tune-flan-t5-deepspeed). In this blog post, we look into how we can integrate DeepSpeed into Amazon SageMaker to allow any practitioners to train those billion parameter size models with a simple API call. Amazon SageMaker managed training allows you to train large language models without having to manage the underlying infrastructure. You can find more information about Amazon SageMaker in the [documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html).

This means we will learn how to fine-tune FLAN-T5 XL & XXL using model parallelism, multiple GPUs, and [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/) on Amazon SageMaker.

## [Notebook](./sagemaker-notebook.ipynb)