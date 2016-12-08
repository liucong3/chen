require 'CELoss'

local L, parent = torch.class('Loss', 'nn.CELoss')

function L:__init()
	parent.__init(self)
end

function L:updateOutput(input, target)
	target = target:narrow(1, 2, target:size(1) - 1)
	return parent.updateOutput(self, input, target)
end

function L:updateGradInput(input, target)
	target = target:narrow(1, 2, target:size(1) - 1)
	return parent.updateGradInput(self, input, target)
end
