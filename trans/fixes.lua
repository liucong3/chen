
require 'nn'

function nn.Container:accUpdateGradParameters(input, gradOutput, lr)
	if self:parameters() then
		self:zeroGradParameters()
		self:accGradParameters(input, gradOutput, 1)
		self:updateParameters(lr)
	end
end

function nn.Module:shareClone()
   error('Module with parameters must implement function "shareClone" to be used in nn.Dynamic')
end

local nn_module_old_type = nn.Module.type
function nn.Module:type(type, ...)
   if not type then return self._type end
   self:nn_module_old_type(type, ...)
   self._type = type
end

function nn.Linear:shareClone()
  local linear = nn.Linear(1, 1, self.bias ~= nil)
  linear.weight = self.weight
  linear.gradWeight = self.gradWeight
  if self.bias ~= nil then
    linear.bias = self.bias
    linear.gradBias = self.gradBias
  end
  return linear
end

function nn.BatchNormalization:shareClone()
  local nOutput = self.running_mean:size(1)
  local batchNorm = nn.BatchNormalization(nOutput, self.eps, self.momentum, self.affine)
  if self.affine then
    batchNorm.weight = self.weight
    batchNorm.bias = self.bias
    batchNorm.gradWeight = self.gradWeight
    batchNorm.gradBias = self.gradBias
  end
  return batchNorm
end

function nn.LookupTable:shareClone()
   local lookup = nn.LookupTable(1, 1)
   lookup.weight = self.weight
   lookup.gradWeight = self.gradWeight
   return lookup
end

--------------------

function appendToTable(table, appended)
    local n = #table
    for i = 1, #appended do
        table[n + i] = appended[i]
    end
end
