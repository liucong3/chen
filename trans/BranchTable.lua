require 'nn'

--[[
BranchTable: branch a single input to multiple outputs

b = nn.BranchTable{3,4,5}
batch_size = 10
input = torch.randn(batch_size,12)
output = b:forward(input)
for _,out in ipairs(output) do
	print(out:size()) -- 10 3   10 4   10 5
end

]]--

local BranchTable, parent = torch.class('nn.BranchTable', 'nn.Module');

function BranchTable:__init(output_sizes)
	self.output_sizes = output_sizes
	self.output = {}
	for i = 1, #output_sizes do
		self.output[i] = torch.Tensor()
	end
	self.gradInput = torch.Tensor()
end

function BranchTable:updateOutput(input)
	local count = 1;
	for i = 1, #self.output_sizes do
		self.output[i]:set(input:narrow(2, count, self.output_sizes[i]))
		count = count + self.output_sizes[i]
	end
	assert(count - 1 == input:size(2), 'Output size(' .. (count - 1) .. ') does not match input:size(2)=' .. input:size(2))
	return self.output
end

function BranchTable:updateGradInput(input, gradOutput)
	--self.gradInput = self.gradInput:typeAs(input)
	self.gradInput:resizeAs(input)
	local count = 1
	for i = 1, #self.output_sizes do
		self.gradInput:narrow(2, count, self.output_sizes[i]):copy(gradOutput[i])
		count = count + self.output_sizes[i]	
	end
	return self.gradInput
end
