import torch
from ..utils.postprocess import postprocess

# Function to compute the accuracy on the validation set
def compute_accuracy(model, valid_loader, criterion, converter, device):
	# Set the model to evaluation mode
	model.eval()

	# Initialize values
	running_loss = 0.0
	correct_predictions = 0
	total_samples = 0

	# Disable gradient computation during validation
	with torch.no_grad():
		# Iterate over the validation set
		for i, (images, labels, labels_len) in enumerate(valid_loader):
			# Convert labels
			label_input, label_len, label_target = converter.train_encode(list(labels))
			# Move to device
			label_input, label_target, images = label_input.to(device), label_target.to(device), images.to(device)
			# Output prediction
			pred = model((images, label_input))	
			# Convert data
			preds, prob = postprocess(pred, converter, None)	
			# Build accuracy list
			acc_list = [(pred == targ) for pred, targ in zip(preds, labels)]
			# Update total samples and correct predictions
			total_samples += len(acc_list)
			correct_predictions += sum(acc_list)
			# Compute the loss
			loss = criterion(pred, label_target, label_len, images.shape[0])
			# Update running loss
			running_loss += loss.item()

	# Compute the average loss
	average_loss = running_loss / len(valid_loader)
	# Calculate the validation accuracy
	validation_accuracy = correct_predictions / total_samples

	# Return average loss and validation accuracy
	return average_loss, validation_accuracy


# Function that trains the mode
def train(model, optimizer, converter, criterion, es, train_loader, valid_loader, num_epochs, device):
	# Initialize training results
	res = {
		'Train Losses': list(),
		'Train Accuracies': list(),
		'Valid Losses': list(),
		'Valid Accuracies': list()
	}
	
	# Iterate over epochs
	for epoch in range(num_epochs):
		# Set model to training mode
		model.train()		

		# Initialize the running loss for this epoch
		running_loss = 0.0		
		correct_predictions = 0
		total_samples = 0	

		# Iterate over the dataloader
		for i, (images, labels, labels_len) in enumerate(train_loader):
			# Zero the gradients for this batch
			optimizer.zero_grad()
			# Convert
			label_input, label_len, label_target = converter.train_encode(labels)
			# Move to device
			label_input, label_target, images = label_input.to(device), label_target.to(device), images.to(device)
			# Forward pass
			pred = model((images, label_input))
			# Convert data
			preds, prob = postprocess(pred, converter, None)	
			# Build accuracy list
			acc_list = [(pred == targ) for pred, targ in zip(preds, labels)]
			# Update total samples and correct predictions
			total_samples += len(acc_list)
			correct_predictions += sum(acc_list)
			# Compute the loss
			loss = criterion(pred, label_target, label_len, images.shape[0])
			# Backpropagation
			loss.backward()
			# Update the model's parameters
			optimizer.step()
			# Update running loss
			running_loss += loss.item()
			
			# Print the training progress for each batch
			if i % 20 == 0:
				print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{i}/{len(train_loader)}] Train Loss: {loss.item():.4f}")

		# Compute the average loss for the epoch
		train_loss = running_loss / len(train_loader)
		# Compute the test accuracy
		train_accuracy = correct_predictions / total_samples
		# Compute validation loss and accuracy
		valid_loss, valid_accuracy = compute_accuracy(model, valid_loader, criterion, converter, device)

		# Print the report
		print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Accuracy {train_accuracy:.4f}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}\n\n")
		# Update results
		res['Train Losses'].append(train_loss)
		res['Train Accuracies'].append(train_accuracy)
		res['Valid Losses'].append(valid_loss)
		res['Valid Accuracies'].append(valid_accuracy)
		# Check early stop
		if es(valid_loss): break
	
	# Print report
	print(f"Finished training. Best validation loss was {es.best:.4f}")
	# Return results
	return res
