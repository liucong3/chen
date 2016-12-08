require 'nn'
require 'CrossEntropyCriterion2'

local L, parent = torch.class('nn.CELoss', 'nn.Criterion')

function L:__init()
	parent.__init(self)
	self.criterion = nn.CrossEntropyCriterion2()
end

function L:updateOutput(input, target)
	if type(input) ~= 'table' then
		self.output = self.criterion:updateOutput(input, target)
		self.gradInput = self.criterion:updateGradInput(input, target)
		self.gradInput = self.gradInput:clone()	
	else
		self.output = 0
		self.gradInput = {}
		for i = 1, #input do
			self.output = self.output + self.criterion:updateOutput(input[i], target:select(1, i):view(-1))
			self.gradInput[i] = self.criterion:updateGradInput(input[i], target:select(1, i):view(-1))
			self.gradInput[i] = self.gradInput[i] / #input
		end
		self.output = self.output / #input
	end
	return self.output
end

function L:updateGradInput(input, target)
	--[[
	if type(input) ~= 'table' then
		self.gradInput = self.criterion:updateGradInput(input, target)
		self.gradInput = self.gradInput:clone()
	else
		self.gradInput = {}
		for i = 1, #input do
			self.gradInput[i] = self.criterion:updateGradInput(input[i], target:select(1, i):view(-1))
			self.gradInput[i] = self.gradInput[i]:clone()
		end
	end
	]]--
	return self.gradInput
end

--[[
require 'Check'
local c = Check()
local input = c.randData({{4,5},{4,5}})
local label = c.randIndex({{2,4}}, {5})[1]
print(input, label)
c.gradCheck(nn.Identity(), nn.CELoss(), input, label, 1e-6)
--[[]]--
