require 'nn'

local L, parent = torch.class('nn.MSELoss', 'nn.Criterion')

function L:__init()
	parent.__init(self)
	self.criterion = nn.MSECriterion()
end

function L:updateOutput(input, target)
	if type(input) ~= 'table' then
		self.output = self.criterion:updateOutput(input, target)
	else
		self.output = 0
		for i = 1, #input do
			self.output = self.output + self.criterion:updateOutput(input[i], target[i])
		end
	end
	return self.output
end

function L:updateGradInput(input, target)
	if type(input) ~= 'table' then
		self.gradInput = self.criterion:updateGradInput(input, target):clone()
	else
		self.gradInput = {}
		for i = 1, #input do
			self.gradInput[i] = self.criterion:updateGradInput(input[i], target[i]):clone()
		end
	end
	return self.gradInput
end

--[[
require 'Check'
local c = Check()
local input = c.randData({{4,5},{4,5}})
local label = c.randData({{4,5},{4,5}})
print(input, label)
c.gradCheck(nn.Identity(), nn.MSELoss(), input, label, 1e-6)
--[[]]--
