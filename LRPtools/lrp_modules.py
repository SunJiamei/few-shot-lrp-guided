import torch
import torch.nn as nn
import torch.nn.functional as F
import LRPtools.utils as util
from methods.backbone import Add as resAdd
from methods.backbone import Flatten as resFlatten

class LRPLayer:
    def _clone_module(self, module):
        # TODO clones should be only computed once during initialization, not for every forward run
        raise NotImplementedError("Abstract base class")

    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):
        raise NotImplementedError("Abstract base class")

class Linear(LRPLayer):
    def _clone_module(self, module):
        clone = nn.Linear(module.in_features, module.out_features)
        # Why can't we set the bias if it's None?
        return clone.to(module.weight.device)

    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):
        ignore_bias = lrp_params.get("ignore_bias", True)
        input_ = module.input[0]
        input_.masked_fill_(input_ == 0, util.RELEVANCE_RECT)
        # ###print(input_)
        # ###print(relevance_input)
        # ###print(relevance_output[0].sum())
        # grad_input = (grad_bias, error, grad_weights)?
        if lrp_method == "zB":
            # todo fuse with alpha_beta code below
            W = module.weight.clone().detach()
            V = torch.nn.functional.relu(W)
            U = -torch.nn.functional.relu(-W)
            L = util.LOWEST * torch.ones_like(input_)
            H = util.HIGHEST * torch.ones_like(input_)

            Z = torch.mm(input_.clone().detach(), W.t()) - torch.mm(L, V.t()) - torch.mm(H,
                                                                        U.t())
            S = util.safe_divide(relevance_output[0], Z)
            R = input_.clone().detach() * torch.mm(S, W) - L * torch.mm(S, V) - H * torch.mm(S, U)
        elif lrp_method == "alpha_beta":
            iself = self._clone_module(module)

            if ignore_bias:
                util.zero_bias(iself)
            iself.weight = nn.Parameter(module.weight.clone().detach())

            # Layer with negative weights and zero bias
            nself = self._clone_module(module)
            if ignore_bias:
                util.zero_bias(nself)
            nself.weight = nn.Parameter(module.weight.clone().detach().clamp(max=0))

            # Layer with positive weights and zero bias
            pself = self._clone_module(module)
            if ignore_bias:
                util.zero_bias(pself)
            pself.weight = nn.Parameter(module.weight.clone().detach().clamp(min=0))

            with torch.enable_grad():
                X = input_.clone().detach().requires_grad_(True)

                if lrp_method == "zB":
                    L = (util.LOWEST * torch.ones_like(X)).requires_grad_(True)
                    H = (util.HIGHEST * torch.ones_like(X)).requires_grad_(True)
                    Z = iself(X) - pself(L) - nself(H)
                    S = util.safe_divide(relevance_output[0].clone().detach(), Z)
                    Z.backward(S)
                    R = X * X.grad + L * L.grad + H * H.grad
                elif lrp_method == "alpha_beta":
                    # See rewritten Eq. 3 in https://www.sciencedirect.com/science/article/pii/S1051200417302385?via%3Dihub#se0120

                    X_pos = input_.clone().detach().clamp(min=0).requires_grad_(
                        True)
                    X_neg = input_.clone().detach().clamp(max=0).requires_grad_(
                        True)
                    # Positive contribution
                    R_pos = lrp_params["alpha"] * (
                            util.lrp_backward(X_pos, pself, relevance_output[0])
                            + util.lrp_backward(X_neg, nself, relevance_output[0])
                    )

                    # Clear gradients
                    util.zero_grad_tensor(X_pos)
                    util.zero_grad_tensor(X_neg)

                    # Negative contribution
                    R_neg = lrp_params["beta"] * (
                            util.lrp_backward(X_neg, pself, relevance_output[0])
                            + util.lrp_backward(X_pos, nself, relevance_output[0])
                    )

                    R = R_pos - R_neg
                else:
                    raise ValueError(f"LRP method {lrp_method} not known.")
        else:
            V = {
                "epsilon": module.weight.clone().detach(),
                "epsilon_IB": module.weight.clone().detach(),
                "z+": torch.nn.functional.relu(module.weight.clone().detach()),
            }[lrp_method]
            # ###print(V.shape)
            Z = torch.mm(input_.clone().detach(), V.t())
            # ###print(Z.shape)
            if ignore_bias:
                # TODO this seems to be not done in iNNvestigate if biases are not ignored.
                Z += util.EPSILON * Z.sign()  # Z.sign() returns -1 or 0 or 1
                Z.masked_fill_(Z == 0, util.EPSILON)
            if not ignore_bias:
                Z += module.bias.clone().detach()
            S = relevance_output[0].clone().detach() / Z
            # ###print(relevance_output[0])
            # ###print(S)
            C = torch.mm(S, V)
            # ###print(C)
            R = input_ * C

        # check_sum(Z.detach(), EPSILON)
        # check_relevance_conservation(R.detach(), relevance_output[0].detach(), module)
        ###print('linear', lrp_method)
        # ###print(module.weight.requires_grad)
        # ###print(relevance_input[0].shape)
        # ###print(relevance_output[0].shape)
        # ###print(input_)
        # ###print(R)
        ###print(R.sum(), R.shape)
        # ###print(len(relevance_input), len(relevance_output)) 3,1
        assert R.shape == input_.shape
        assert not torch.isnan(R.sum())
        assert not torch.isinf(R.sum())
        module.zero_grad()
        if len(relevance_input) == 3:
            assert relevance_input[0].shape == module.bias.shape
            assert relevance_input[1].shape == R.shape
            # TODO why is this tranposed? Is it the gradient of the weight in the first place?
            assert relevance_input[2].shape == module.weight.t().shape
            # ###print(relevance_input[0].shape, R.shape, relevance_input[2].shape)
            return relevance_input[0], R, relevance_input[2]
        elif len(relevance_input) == 2:
            # No biases
            assert relevance_input[0].shape == R.shape
            # TODO why is this tranposed? Is it the gradient of the weight in the first place? yes
            assert relevance_input[1].shape == module.weight.t().shape

            return R, relevance_input[1]

class ReLU(LRPLayer):
    # def propagate_relevance(self, module, relevance_input, relevance_output,
    #                         lrp_method, lrp_params=None, additional_relevance=None):
    #     # ###print('relu', lrp_method)
    #     # ###print(relevance_input[0].shape)
    #     # ###print(relevance_output[0].shape)
    #     # ###print(len(relevance_output))
    #     # ###print('relu',relevance_output[0].sum())
    #     return relevance_output
    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):
        if lrp_method == 'identity':
            return(relevance_output[0],)
        else:
            v_input = module.input[0]
            mask = torch.where(v_input>0, torch.full_like(v_input,1), torch.full_like(v_input, 0))

            # ###print(relevance_input[0].shape)
            # ###print(relevance_output[0].shape)
            # ###print(relevance_output[0].sum())
            # ###print(v_input.shape)
            # ###print(len(relevance_input))
            R = relevance_output[0].clone().detach()
            R = R*mask
            # ###print(len(relevance_output), len(relevance_output))
            ###print('relu', lrp_method, R.sum())
            assert not torch.isnan(R.sum())
            assert not torch.isinf(R.sum())
            return (R,)

class PosNetConv(nn.Module):

    def _clone_module(self, module):
        clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                          **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
        super(PosNetConv, self).__init__()

        self.posconv = self._clone_module(conv)
        self.posconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(min=0),requires_grad=False)

        self.negconv = self._clone_module(conv)
        self.negconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(max=0),requires_grad=False)

        if ignorebias == True:
            self.posconv.bias = None
            self.negconv.bias = None
        else:
            if conv.bias is not None:
                self.posconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(min=0),requires_grad=False)
                self.negconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(max=0),requires_grad=False)


    def forward(self, x):
        vp = self.posconv(torch.clamp(x, min=0))
        vn = self.negconv(torch.clamp(x, max=0))
        return vp + vn

class NegNetConv(nn.Module):

    def _clone_module(self, module):
        clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                          **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)

    def __init__(self, conv, ignorebias):
        super(NegNetConv, self).__init__()

        self.posconv = self._clone_module(conv)
        self.posconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(min=0),requires_grad=False)

        self.negconv = self._clone_module(conv)
        self.negconv.weight = torch.nn.Parameter(conv.weight.data.clone().clamp(max=0),requires_grad=False)

        if ignorebias == True:
            self.posconv.bias = None
            self.negconv.bias = None
        else:
            if conv.bias is not None:
                self.posconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(min=0),requires_grad=False)
                self.negconv.bias = torch.nn.Parameter(conv.bias.data.clone().clamp(max=0),requires_grad=False)


    def forward(self, x):
        vp = self.posconv(torch.clamp(x, max=0))
        vn = self.negconv(torch.clamp(x, min=0))
        return vp + vn

class lrp_Conv2d_beta0(LRPLayer):
    def _clone_module(self, module):
        clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                          **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)
        # copy deepcopy ?

    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):

        ignorebias = False
        if lrp_params is not None:
            if 'conv2d_ignorebias' in lrp_params:
                ignorebias = lrp_params['conv2d_ignorebias']

        # print('at lrp conv2d default')
        assert (isinstance(module, nn.Conv2d))

        # input relevance for list relevance_input is: (x, w, bias if exists)

        input_ = module.input[0]  # [0]? tuple? what are the other inputs?
        pnconv = PosNetConv(module, ignorebias)

        X = input_.clone().detach().requires_grad_(True)

        R = util.lrp_backward(_input=X, layer=pnconv, relevance_output=relevance_output[0])

        print('conv2d reldiff', torch.sum(relevance_output[0]) - torch.sum(R))

class Conv2d(LRPLayer):
    #@profile()
    def _clone_module(self, module):

        clone = nn.Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                         **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'groups']})
        return clone.to(module.weight.device)
    #@profile()
    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):

        ignore_bias = lrp_params.get("ignore_bias", True)
        input_ = module.input[0]
        # for i in range(len(relevance_input)):
        #     ###print(i, relevance_input[i].shape)
        # ###print(module.weight.shape, module.bias.shape, module.padding)
        # ###print(module.weight.requires_grad)

        if lrp_method in ["zB", "alpha_beta"]:
            # Layer with same weights but zero bias
            iself = self._clone_module(module)
            if ignore_bias:
                util.zero_bias(iself)
            iself.weight = nn.Parameter(module.weight.clone().detach())

            # Layer with negative weights and zero bias
            nself = self._clone_module(module)
            if ignore_bias:
                util.zero_bias(nself)
            nself.weight = nn.Parameter(module.weight.clone().detach().clamp(max=0))

            # Layer with positive weights and zero bias
            pself = self._clone_module(module)
            if ignore_bias:
                util.zero_bias(pself)
            pself.weight = nn.Parameter(module.weight.clone().detach().clamp(min=0))
            pnconv = PosNetConv(module, ignore_bias)
            nnconv = NegNetConv(module, ignore_bias)
            with torch.enable_grad():
                X = input_.clone().detach().requires_grad_(True)

                if lrp_method == "zB":
                    L = (util.LOWEST * torch.ones_like(X)).requires_grad_(True)
                    H = (util.HIGHEST * torch.ones_like(X)).requires_grad_(True)
                    Z = iself(X) - pself(L) - nself(H)
                    S = util.safe_divide(relevance_output[0].clone().detach(), Z)
                    Z.backward(S)
                    R = X * X.grad + L * L.grad + H * H.grad
                elif lrp_method == "alpha_beta":
                    # See rewritten Eq. 3 in https://www.sciencedirect.com/science/article/pii/S1051200417302385?via%3Dihub#se0120

                    X = input_.clone().detach().requires_grad_(True)

                    # Positive contribution
                    R_pos = lrp_params["alpha"] * (
                            util.lrp_backward(_input=X, layer=pnconv, relevance_output=relevance_output[0])
                    )
                    # Clear gradients
                    util.zero_grad_tensor(X)

                    # Negative contribution
                    R_neg = lrp_params["beta"] * (
                        util.lrp_backward(_input=X, layer=nnconv, relevance_output=relevance_output[0])
                    )
                    R = R_pos - R_neg

                    del pnconv
                    del nnconv
                else:
                    raise ValueError(f"LRP method {lrp_method} not known.")
        elif lrp_method == "z+":
            pself = self._clone_module(module)
            if ignore_bias:
                util.zero_bias(pself)
            pself.weight = nn.Parameter(F.relu(module.weight.clone().detach()))

            with torch.enable_grad():
                X = input_.clone().detach().requires_grad_(True)
                R = util.lrp_backward(X, pself, relevance_output[0])
        elif lrp_method == "flat":
            raise NotImplementedError
        elif lrp_method == "epsilon":
            raise NotImplementedError
        elif lrp_method == "gamma":
            raise NotImplementedError
        else:
            raise ValueError(f"LRP method {lrp_method} not known.")

        # check_sum(Z.detach(), EPSILON)
        # check_relevance_conservation(R.detach(), relevance_output[0].detach(), module)
        # ###print('conv2d', lrp_method)
        # ###print(relevance_input[0].shape)
        # ###print(relevance_output[0].shape)
        # ###print(R.shape)
        # ###print(relevance_input[0].shape, relevance_input[1].shape, relevance_input[2].shape)
        ###print('conv2d',relevance_output[0].sum(),R.sum())
        # ###print(len(relevance_input))
        # ###print(relevance_input[0].shape, relevance_input[1].shape, R.shape)
        assert R.shape == input_.shape
        assert not torch.isnan(R.sum())
        assert not torch.isinf(R.sum())
        module.zero_grad()
        if len(relevance_input) == 3:
            assert relevance_input[0].shape == R.shape
            assert relevance_input[1].shape == module.weight.shape
            if module.bias is None:
                assert relevance_input[2] is None
            else:
                assert relevance_input[2].shape == module.bias.shape
            # ###print('conv2d', relevance_output[0].sum(), R.sum())
            torch.cuda.empty_cache()
            return R, relevance_input[1], relevance_input[2]
        elif len(relevance_input) == 2:
            # No biases
            assert relevance_input[0].shape == R.shape
            assert relevance_input[1].shape == module.weight.shape
            # ###print('conv2d', relevance_output[0].sum(), R.sum())
            torch.cuda.empty_cache()
            return R, relevance_input[1]

class Pool2d(LRPLayer):
    def _clone_module(self, module):
        if type(module) == nn.MaxPool2d:
            clone = nn.MaxPool2d(module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'dilation', 'return_indices', 'ceil_mode']})
        elif type(module) == nn.AvgPool2d:
            clone = nn.AvgPool2d(module.kernel_size, **{attr: getattr(module, attr) for attr in ['stride', 'padding', 'count_include_pad', 'ceil_mode']})
        else:
            raise ValueError(type(module))
        return clone

    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):
        # ###print('pool2d', lrp_method)
        # ###print(relevance_input[0].shape)
        # ###print(relevance_output[0].shape)
        input_ = module.input[0]
        module_clone = self._clone_module(module)
        with torch.enable_grad():
            X = input_.clone().detach().requires_grad_(True)
            Z = module_clone(X)
            S = util.safe_divide(relevance_output[0].clone().detach(), Z)
            Z.backward(S)
            R = X * X.grad

        # check_relevance_conservation(R.detach(), relevance_output[0].detach(), module)
        # ###print(R.shape)
        ###print('pool', R.sum(), R.shape, relevance_input[0].shape)
        assert not torch.isnan(R.sum())
        assert not torch.isinf(R.sum())
        module.zero_grad()
        return (R, )

class BatchNorm2d(LRPLayer):
    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):
        # todo how is batchnorm handled by default in innvestigate?
        ###print('batchnorm2d', lrp_method)
        # ###print(len(relevance_input), len(relevance_output), relevance_output[0].shape)
        # for i in range(len(relevance_input)):
            # ###print(relevance_input[i].shape)
        # ###print(module._buffers.keys())
        if lrp_method == 'identity':
            R = relevance_output[0]

        else:
            input_ = module.input[0]
            mean = module._buffers['running_mean']
            var = module._buffers['running_var']
            gamma = module._parameters['weight']
            beta = module._parameters['bias']

            w = (gamma / torch.sqrt(var + module.eps))[:, None, None]
            b = (beta - (mean * gamma) / torch.sqrt(var + module.eps))[:, None, None]
            xw = input_ * w
            # ###print(w.shape, b.shape, xw.shape)
            # ###print(relevance_output[0].shape)

            R = util.safe_divide(torch.abs(xw), (torch.abs(xw) + torch.abs(b))) * \
                relevance_output[0]
            # ###print(len(relevance_output))
        ###print('batchnorm2d',relevance_output[0].sum(), R.sum())
        assert not torch.isnan(R.sum())
        assert not torch.isinf(R.sum())
        assert R.sum() !=0
        module.zero_grad()
        return R, relevance_input[1], relevance_input[2]

class BatchNorm1d(LRPLayer):
    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):
        # todo how is batchnorm handled by default in innvestigate?
        ###print('batchnorm2d', lrp_method)
        # ###print(len(relevance_input), len(relevance_output), relevance_output[0].shape)
        # for i in range(len(relevance_input)):
            # ###print(relevance_input[i].shape)
        # ###print(module._buffers.keys())
        if lrp_method == 'identity':
            R = relevance_output[0]

        else:
            input_ = module.input[0]
            mean = module._buffers['running_mean']
            var = module._buffers['running_var']
            gamma = module._parameters['weight']
            beta = module._parameters['bias']

            w = (gamma / torch.sqrt(var + module.eps))[:, None, None]
            b = (beta - (mean * gamma) / torch.sqrt(var + module.eps))[:, None, None]
            xw = input_ * w
            # ###print(w.shape, b.shape, xw.shape)
            # ###print(relevance_output[0].shape)

            R = util.safe_divide(torch.abs(xw), (torch.abs(xw) + torch.abs(b))) * \
                relevance_output[0]
            # ###print(len(relevance_output))
        ###print('batchnorm2d',relevance_output[0].sum(), R.sum())
        assert not torch.isnan(R.sum())
        assert not torch.isinf(R.sum())
        assert R.sum() !=0
        module.zero_grad()
        return R, relevance_input[1], relevance_input[2]

class Dropout(LRPLayer):
    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):
        # TODO how to handle this?
        # print(relevance_output[0].shape)
        # print(((relevance_output[0] - relevance_input[0]).abs().max() < 1e-7))
        assert ((relevance_output[0] - relevance_input[0]).abs().max() < 1e-7).cpu().item() == 1
        module.zero_grad()
        return relevance_input

class Add(LRPLayer):
    def _clone_module(self, module):
        clone = resAdd()
        # Why can't we set the bias if it's None?
        return clone.to(module.weight.device)
    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):
        # ###print(module.input[0].shape, module.input[1].shape)
        # ###print(len(relevance_input)) #2
        # ###print(len(relevance_output)) #1
        # ###print(relevance_input[0].shape)   # same as input[0]
        # ###print(relevance_input[1].shape)   # same as input[1]
        input_1, input_2 = module.input[0].detach(), module.input[1].detach()
        # ###print(input_1.sum(), input_2.sum())
        out = input_1 + input_2
        mask = out == 0
        out_mask = torch.zeros_like(out).masked_fill_(mask, 0.5)

        out += util.EPSILON * out.sign()
        rele_out = relevance_output[0].clone().detach()
        R_mask = rele_out * out_mask
        R1 =  rele_out * input_1 / out
        R2 = rele_out * input_2 / out
        R1[R1!=R1] = 0
        R2[R2!=R2] = 0
        R1 += R_mask
        R2 += R_mask
        ###print('add',R1.sum(), R2.sum(), relevance_output[0].sum())
        assert not torch.isnan(R1.sum())
        assert not torch.isnan(R2.sum())
        assert not torch.isinf(R1.sum())
        assert not torch.isinf(R2.sum())
        return R1, R2

class Flatten(LRPLayer):
    # def propagate_relevance(self, module, relevance_input, relevance_output,
    #                         lrp_method, lrp_params=None, additional_relevance=None):
    #     # ###print('relu', lrp_method)
    #     # ###print(relevance_input[0].shape)
    #     # ###print(relevance_output[0].shape)
    #     # ###print(len(relevance_output))
    #     # ###print('relu',relevance_output[0].sum())
    #     return relevance_output
    def propagate_relevance(self, module, relevance_input, relevance_output,
                            lrp_method, lrp_params=None):

        v_input = module.input[0]
        v_size = v_input.size()
        R = relevance_output[0].clone().detach()
        R = R.view(v_size)
        # ###print(len(relevance_output), len(relevance_output))
        ###print('relu', lrp_method, R.sum())
        assert not torch.isnan(R.sum())
        assert not torch.isinf(R.sum())
        return (R,)



def compute_lrp_sum(sum_output, sum_input, relevance_sum_output,dim=-1):
    #this function will return the relevance of the sum_input
    # ###print(sum_output)
    # ###print(torch.sum(sum_input,dim=dim))
    # ###print(sum_output, torch.sum(sum_input, dim=dim))
    assert (sum_output == torch.sum(sum_input,dim=dim)).all()
    fea_dim = sum_input.size()[-1]
    # ###print(relevance_sum_output.unsqueeze(-1).repeat(1,1,1,fea_dim).shape)
    # ###print(sum_input.shape)
    relevance = relevance_sum_output.unsqueeze(-1).repeat(1,1,1,fea_dim)
    # ###print(relevance)
    out = sum_output.unsqueeze(-1).repeat(1,1,1,fea_dim)
    mask = out == 0
    out.masked_fill_(mask, 1 / fea_dim)
    relevance_sum_input = relevance * sum_input / (out + util.EPSILON * out.sign())
    return relevance_sum_input

def compute_lrp_mean(mean_output, mean_input, relevance_mean_output,dim=-1):
    #this function will return the relevance of the mean_input
    assert (mean_output == torch.mean(mean_input,dim=dim)).all()
    fea_dim = mean_input.size()[-1]
    input_dim = len(mean_input.shape)
    repeat_param = [1]*input_dim
    repeat_param[-1] *= fea_dim
    # ###print(repeat_param)
    relevance = relevance_mean_output.unsqueeze(-1).repeat(repeat_param)
    out = mean_input.sum(dim=dim).unsqueeze(-1).repeat(repeat_param)
    mask = out == 0
    # input = mean_input / fea_dim
    out.masked_fill_(mask, 1/fea_dim)
    relevance_mean_input = relevance * mean_input / (out + util.EPSILON * out.sign())
    return relevance_mean_input

def get_lrp_module(module):
    try:
        lrp_module_class = {
            nn.Linear: Linear,
            nn.ReLU: ReLU,
            nn.Conv2d: Conv2d,
            nn.MaxPool2d: Pool2d,
            nn.AvgPool2d: Pool2d,  # TODO
            nn.BatchNorm2d: BatchNorm2d,
            nn.BatchNorm1d: BatchNorm1d,
            nn.Dropout: Dropout,
            nn.Dropout2d: Dropout,
            resFlatten: Flatten,
            resAdd:Add,
        }[type(module)]
    except KeyError:
        ###print(type(module))
        raise ValueError("Layer type {} not known.".format(type(module)))

    lrp_module = lrp_module_class()
    return lrp_module






