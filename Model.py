import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    def __init__(self, config, datamanager):
        super(Model, self).__init__()
        self.config = config

        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=torch.tensor(datamanager.get_word_embedding()),
            freeze=False,
            padding_idx=0
        )
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=config.word_vec_dim,
                    out_channels=config.filter_num,
                    kernel_size=kernel_size,
                    bias=True
                ),
                nn.Tanh(),
                nn.MaxPool1d(kernel_size=config.max_sen_len - kernel_size + 1)
            )
            for kernel_size in config.kernel_size_list
        ])
        self.lstm = nn.LSTM(
            input_size=config.word_vec_dim,
            hidden_size=config.word_vec_dim,
            num_layers=config.num_layer,
            batch_first=True,
            bidirectional=True
        )
        self.w = nn.Parameter(torch.randn(config.word_vec_dim))
        self.fc = nn.Sequential(
            nn.Linear(
                in_features=250,
                out_features=50
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=50,
                out_features=2
            )
        )
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(
            params=self.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )

    def lstm_layer(self, x, lengths):
        batch_size = x.size()[0]
        x = pack_padded_sequence(
            input=x,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )
        h, (_, _) = self.lstm(x)
        h, _ = pad_packed_sequence(
            sequence=h,
            batch_first=True,
            total_length=self.config.max_sen_len
        )
        h = h.view(-1, self.config.max_sen_len, 2, self.config.word_vec_dim)
        h = torch.sum(h, dim=2)

        m = torch.tanh(h)
        alpha = F.softmax(torch.bmm(m, self.w.repeat(batch_size, 1, 1).transpose(1, 2)), dim=0)
        r = torch.bmm(alpha.transpose(1, 2), h)
        h_star = torch.tanh(r)
        h_star = h_star.permute(0, 2, 1)
        return h_star

    def forward(self, sentences, lengths):
        x = self.word_embedding(sentences)

        conv_out = [conv(x.permute(0, 2, 1)) for conv in self.conv]
        lstm_out = self.lstm_layer(x, lengths)

        out = conv_out
        out.append(lstm_out)
        out = torch.cat(out, dim=1)
        out = out.view(-1, out.size(1))

        out = self.fc(out)

        return out

    def step(self, data):
        scores = self.forward(data[0], data[1])
        loss = self.loss(scores, data[2])
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.data.tolist()

    @torch.no_grad()
    def evaluate(self, data):
        scores = self.forward(data[0], data[1])
        predicted_result = torch.argmax(scores, dim=-1)
        gt_result = data[2]
        correct_num = torch.sum(predicted_result == gt_result)
        return correct_num.data.tolist(), len(predicted_result)
