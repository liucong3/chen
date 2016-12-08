require 'nn'

--[[
MergeTable：merge multiple inputs into a single output

m = nn.MergeTable(3)
batch_size = 10
input = { torch.randn(batch_size, 3), torch.randn(batch_size, 4), torch.randn(batch_size, 5) }
output = m:forward(input)
print(output:size()) -- 10 12

]]--

local MergeTable, parent = torch.class('nn.MergeTable', 'nn.Module');

function MergeTable:__init(num_input)
	self.output = torch.Tensor()
	self.gradInput = {}
	for i = 1, num_input do
		self.gradInput[i] = torch.Tensor()
	end
end

function MergeTable:updateOutput(input)
	local output_size = 0;
	for i = 1, #input do
		output_size = output_size + input[i]:size(2)
	end
	--self.output = self.output:typeAs(input[1])
	self.output:resize(input[1]:size(1), output_size)
	local count = 1
	for i = 1, #input do
		self.output:narrow(2, count, input[i]:size(2)):copy(input[i])
		count = count + input[i]:size(2)
	end
	return self.output
end

function MergeTable:updateGradInput(input, gradOutput)
	local count = 1
	for i = 1, #input do
		self.gradInput[i]:set(gradOutput:narrow(2, count, input[i]:size(2)))
		count = count + input[i]:size(2)	
	end
	return self.gradInput
end

