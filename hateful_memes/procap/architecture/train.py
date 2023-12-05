import os

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from dataset import MultiModalData
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

MLFLOW_TRACKING_URI = "/home2/faculty/mgalkowski/memes_analysis/mlflow_data"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("hateful_memes")


def bce_for_loss(logits, labels):
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_auc_score(logits, label):
    bz = logits.shape[0]
    logits = logits.cpu().numpy()
    label = label.cpu().numpy()
    auc = roc_auc_score(label, logits, average="weighted") * bz
    return auc


def compute_score(logits, labels):
    # print (logits,logits.shape)
    logits = torch.max(logits, 1)[1]
    # print (logits)
    one_hot = torch.zeros(*labels.size()).cuda()
    one_hot.scatter_(1, logits.view(-1, 1), 1)
    score = one_hot * labels
    return score.sum().float()


def compute_scaler_score(logits, labels):
    logits = torch.max(logits, 1)[1]
    labels = labels.squeeze(-1)
    score = (logits == labels).int()
    return score.sum().float()


def log_hyperpara(logger, opt):
    dic = vars(opt)
    for k, v in dic.items():
        logger.write(k + " : " + str(v))


def train_for_epoch(opt, model, train_loader, test_loader):
    with mlflow.start_run(
        run_name=f"{opt.MODEL_NAME.replace('-', '_')}_{opt.SAVE_NUM}"
    ):
        mlflow.log_params(vars(opt))

        if opt.SAVE:
            model_path = os.path.join(
                opt.MODEL_PATH, "_".join([str(opt.SEED), opt.DATASET])
            )
            if not os.path.exists(model_path):
                os.mkdir(model_path)

        # initialization of logger
        log_path = os.path.join(opt.LOG_PATH)
        if not os.path.exists(log_path):
            os.mkdir(log_path)

        logger = utils.Logger(
            os.path.join(log_path, opt.SAVE_NUM + opt.MODEL_NAME + ".txt")
        )
        log_hyperpara(logger, opt)

        logger.write(
            "Length of training set: %d, length of testing set: %d"
            % (len(train_loader.dataset), len(test_loader.dataset))
        )
        logger.write("Max length of sentences: %d" % (model.max_length))

        # initialization of optimizer
        params = {}
        for n, p in model.named_parameters():
            if opt.FIX_LAYERS > 0:
                if "encoder.layer" in n:
                    try:
                        layer_num = int(n[n.find("encoder.layer") + 14 :].split(".")[0])
                    except Exception as e:
                        print(n)
                        print(e)
                        raise Exception("")
                    if layer_num >= opt.FIX_LAYERS:
                        print("yes", n)
                        params[n] = p
                    else:
                        print("no ", n)
                elif "embeddings" in n:
                    print("no ", n)
                else:
                    print("yes", n)
                    params[n] = p
            else:
                params[n] = p
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in params.items() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": opt.WEIGHT_DECAY,
            },
            {
                "params": [
                    p for n, p in params.items() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optim = AdamW(
            optimizer_grouped_parameters,
            lr=opt.LR_RATE,
            eps=opt.EPS,
        )

        num_training_steps = len(train_loader) * opt.EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        # start training
        record_auc = []
        record_acc = []
        for epoch in range(opt.EPOCHS):
            model.train(True)
            total_loss = 0.0
            scores = 0.0
            for i, batch in enumerate(train_loader):
                # break
                # label = batch["label"].float().cuda().view(-1, 1)
                target = batch["target"].cuda()

                if opt.USE_DEMO:
                    text = batch["prompt_all_text"]
                else:
                    text = batch["test_all_text"]  # without demonstrations

                logits = model(text)

                loss = bce_for_loss(logits, target)
                batch_score = compute_score(logits, target)
                scores += batch_score

                # print("Epoch:", epoch, "Iteration:", i, loss.item())  # , batch_score)
                loss.backward()
                optim.step()
                scheduler.step()
                optim.zero_grad()

                total_loss += loss
            print("Epoch:", epoch + 1, "Loss:", total_loss.item() / len(train_loader))
            mlflow.log_metric(
                "train_loss", total_loss.item() / len(train_loader), step=epoch + 1
            )

            model.train(False)
            scores /= len(train_loader.dataset)
            if opt.USE_DEMO and opt.MULTI_QUERY:
                eval_acc, eval_auc = eval_multi_model(opt, model)
            else:
                eval_acc, eval_auc = eval_model(opt, model, test_loader)
            mlflow.log_metric("eval_acc", eval_acc, step=epoch + 1)
            mlflow.log_metric("eval_auc", eval_auc, step=epoch + 1)
            mlflow.log_metric("train_acc", scores * 100.0, step=epoch + 1)

            record_auc.append(eval_auc)
            record_acc.append(eval_acc)

            logger.write("Epoch %d" % (epoch + 1))
            logger.write(
                "\ttrain_loss: %.2f, accuracy: %.2f" % (total_loss, scores * 100.0)
            )
            logger.write(
                "\tevaluation auc: %.2f, accuracy: %.2f" % (eval_auc, eval_acc)
            )

        max_idx = sorted(
            range(len(record_auc)),
            key=lambda k: record_auc[k] + record_acc[k],
            reverse=True,
        )[0]
        logger.write("Maximum epoch: %d" % (max_idx))
        logger.write(
            "\tevaluation auc: %.2f, accuracy: %.2f"
            % (record_auc[max_idx], record_acc[max_idx])
        )
        if opt.SAVE:
            torch.save(
                model.state_dict(),
                os.path.join(model_path, opt.SAVE_NUM + opt.MODEL_NAME + ".pth"),
            )
            mlflow.pytorch.log_model(
                model, f"{opt.MODEL_NAME.replace('-', '_')}_{opt.SAVE_NUM}"
            )


def eval_model(opt, model, test_loader):
    scores = 0.0
    auc = 0.0
    len_data = len(test_loader.dataset)
    print("Length of test set:", len_data)
    total_logits = []
    total_labels = []
    for _, batch in enumerate(test_loader):
        with torch.no_grad():
            label = batch["label"].float().cuda().view(-1, 1)
            target = batch["target"].cuda()
            # img = batch["img"]

            if opt.USE_DEMO:
                text = batch["prompt_all_text"]
            else:
                text = batch["test_all_text"]  # without demonstrations

            logits = model(text)
            batch_score = compute_score(logits, target)
            scores += batch_score
            norm_logits = F.softmax(logits, dim=-1)[:, 1].unsqueeze(-1)
            # bz = target.shape[0]
            total_logits.append(norm_logits)
            total_labels.append(label)
    total_logits = torch.cat(total_logits, dim=0)
    total_labels = torch.cat(total_labels, dim=0)
    print(total_logits.shape, total_labels.shape)
    auc = compute_auc_score(total_logits, total_labels)
    # print (auc)
    return scores * 100.0 / len_data, auc * 100.0 / len_data


def eval_multi_model(opt, model):
    num_queries = opt.NUM_QUERIES
    labels_record = {}
    logits_record = {}
    prob_record = {}
    for k in range(num_queries):
        test_set = MultiModalData(opt, "test")
        test_loader = DataLoader(test_set, opt.BATCH_SIZE, shuffle=False, num_workers=2)
        len_data = len(test_loader.dataset)
        print("Length of test set:", len_data, "Query:", k)
        for i, batch in enumerate(test_loader):
            with torch.no_grad():
                label = batch["label"].float().cuda().view(-1, 1)
                img = batch["img"]
                target = batch["target"].cuda()
                text = batch["prompt_all_text"]
                logits = model(text)
                norm_prob = F.softmax(logits, dim=-1)
                norm_logits = norm_prob[:, 1].unsqueeze(-1)

                bz = target.shape[0]
                for j in range(bz):
                    cur_img = img[j]
                    cur_logits = norm_logits[j : j + 1]
                    # should normalize to the same scale
                    cur_prob = norm_prob[j : j + 1]
                    if k == 0:
                        cur_label = label[j : j + 1]
                        labels_record[cur_img] = cur_label
                        logits_record[cur_img] = cur_logits
                        prob_record[cur_img] = cur_prob
                    else:
                        logits_record[cur_img] += cur_logits
                        prob_record[cur_img] += cur_prob
    labels = []
    logits = []
    probs = []
    for name in labels_record.keys():
        labels.append(labels_record[name])
        logits.append(logits_record[name] / num_queries)
        probs.append(prob_record[name] / num_queries)

    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)

    scores = compute_scaler_score(probs, labels)
    auc = compute_auc_score(logits, labels)
    return scores * 100.0 / len_data, auc * 100.0 / len_data
