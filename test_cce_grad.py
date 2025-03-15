import torch
import copy

from cut_cross_entropy import linear_cross_entropy
torch.manual_seed(42)

model = torch.nn.Linear(128, 128, dtype=torch.bfloat16).cuda()
classifier_weights = torch.nn.Linear(128, 32, dtype=torch.bfloat16).cuda()

model_c = copy.deepcopy(model)
classifier_weights_c = copy.deepcopy(classifier_weights)

input = torch.randn(8192, 128, dtype=torch.bfloat16).cuda()
input.requires_grad = True
input.retain_grad()

input_c = input.clone()

labels = torch.randint(0, 32, (8192,)).cuda()
labels_c = labels.clone()

embeddings = model(input)

shift_embeddings = embeddings[:-1, :]
shift_labels = labels[1:]

manual_shift_loss = linear_cross_entropy(shift_embeddings, classifier_weights.weight, shift_labels)
manual_shift_loss.backward()

embeddings_c = model_c(input_c)
logits = classifier_weights_c(embeddings_c)
shift_logits_c = logits[:-1, :]
shift_labels_c = labels_c[1:]
manual_shift_loss_c = torch.nn.functional.cross_entropy(shift_logits_c, shift_labels_c)
manual_shift_loss_c.backward()

print(f"model grad: {model.weight.grad.mean()}")
print(f"model_c grad: {model_c.weight.grad.mean()}")

print(f"classifier_weights grad: {classifier_weights.weight.grad.mean()}")
print(f"classifier_weights_c grad: {classifier_weights_c.weight.grad.mean()}")