function torch.ClTensor:uniform(a, b)
   if a == nil then
      a = 0
   end
   if b == nil then
      b = 1
   end
   self:copy(torch.FloatTensor(self:size()):uniform(a, b))
   return self
end

local randn0 = torch.randn
local rand0 = torch.rand

torch.randn = function(...)
  --print('torch.randn')
  local tensorType = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  local tensor = randn0(...)
  torch.setdefaulttensortype(tensorType)
  return tensor
end

torch.rand = function(...)
  --print('torch.rand')
  local tensorType = torch.getdefaulttensortype()
  torch.setdefaulttensortype('torch.FloatTensor')
  local tensor = rand0(...)
  torch.setdefaulttensortype(tensorType)
  return tensor
end
