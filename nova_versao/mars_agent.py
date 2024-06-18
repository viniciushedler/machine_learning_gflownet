

class MARSAgent:
    def __init__(self, args, envs):
        self.model = make_mlp([args.horizon * args.ndim] +
                              [args.n_hid] * args.n_layers +
                              [args.ndim*2])
        self.model.to(args.dev)
        self.dataset = []
        self.dataset_max = args.n_dataset_pts
        self.mbsize = args.mbsize
        self.envs = envs
        self.batch = [i.reset() for i in envs] # The N MCMC chains
        self.ndim = args.ndim
        self.bufsize = args.bufsize

    def parameters(self):
        return self.model.parameters()

    def sample_many(self, mbsize, all_visited):
        s = torch.cat([tf([i[0]]) for i in self.batch])
        r = torch.cat([tf([i[1]]) for i in self.batch])
        with torch.no_grad(): logits = self.model(s)
        pi = SplitCategorical(self.ndim, logits=logits)
        a = pi.sample()
        q_xpx = torch.exp(pi.log_prob(a))
        steps = [self.envs[j].step(a[j].item(), s=self.batch[j][2]) for j in range(len(self.envs))]
        sp = torch.cat([tf([i[0]]) for i in steps])
        rp = torch.cat([tf([i[1]]) for i in steps])
        with torch.no_grad(): logits_sp = self.model(sp)
        reverse_a = tl([i[3] for i in steps])
        pi_sp = SplitCategorical(self.ndim, logits=logits_sp)
        q_xxp = torch.exp(pi.log_prob(reverse_a))
        # This is the correct MH acceptance ratio:
        #A = (rp * q_xxp) / (r * q_xpx + 1e-6)

        # But the paper suggests to use this ratio, for reasons poorly
        # explained... it does seem to actually work better? but still
        # diverges sometimes. Idk
        A = rp / r
        U = torch.rand(self.bufsize)
        for j in range(self.bufsize):
            if A[j] > U[j]: # Accept
                self.batch[j] = (sp[j].numpy(), rp[j].item(), steps[j][2])
                all_visited.append(tuple(steps[j][2]))
            # Added `or U[j] < 0.05` for stability in these toy settings
            if rp[j] > r[j] or U[j] < 0.05: # Add to dataset
                self.dataset.append((s[j].unsqueeze(0), a[j].unsqueeze(0)))
        return [] # agent is stateful, no need to return minibatch data


    def learn_from(self, i, data):
        if not i % 20 and len(self.dataset) > self.dataset_max:
            self.dataset = self.dataset[-self.dataset_max:]
        if len(self.dataset) < self.mbsize:
            return None
        idxs = np.random.randint(0, len(self.dataset), self.mbsize)
        s, a = map(torch.cat, zip(*[self.dataset[i] for i in idxs]))
        logits = self.model(s)
        pi = SplitCategorical(self.ndim, logits=logits)
        q_xxp = pi.log_prob(a)
        loss = -q_xxp.mean()+np.log(0.5)
        # loss_p = loss  - pi.entropy().mean() * 0.1 # no, the entropy wasn't there in the paper
        return loss, pi.entropy().mean()