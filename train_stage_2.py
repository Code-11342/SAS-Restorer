from dependencies import *
from utils import *
from metric import MetricManager
from model import DilatedAutoEncoder, Gendis, ModelManager
from dataset import CompleteDataset3DFixMask, FlawDataset3DWithMask

random_seed = 42
setup_random_seed(seed=random_seed)

# cli
arg_parser = ArgumentParser()
# basic option
arg_parser.add_argument("--model_name", type=str, default="sas_model")
# dataset option
arg_parser.add_argument("--com_dataset_dir", type=str, default="../dataset_dir/complete_dataset")
arg_parser.add_argument("--flaw_dataset_dir", type=str, default="../dataset_dir/flaw_dataset")
# save option
arg_parser.add_argument("--save_dir", type=str, default="../save_dir")
# train option
arg_parser.add_argument("--log_interval", type=int, default=1)
arg_parser.add_argument("--sample_interval", type=int, default=1)
arg_parser.add_argument("--save_interval", type=int, default=1)
arg_parser.add_argument("--batch_size", type=int, default=1)
# parse args
args = arg_parser.parse_args()

# paths
model_name = args.model_name
com_dataset_dir = args.com_dataset_dir
flaw_dataset_dir = args.flaw_dataset_dir
save_dir = args.save_dir
model_dir = f"{save_dir}/{model_name}"
log_dir = f"{model_dir}/log_dir"

# dataset dir
# train
train_src_data_dir=f"{com_dataset_dir}/train/image/*"
train_dst_data_dir=f"{flaw_dataset_dir}/train/image/*"
# val
val_src_data_dir=f"{com_dataset_dir}/val/image/*"
val_dst_data_dir=f"{flaw_dataset_dir}/val/image/*"
# sample
sample_src_data_dir=f"{com_dataset_dir}/sample/gt_image/*"
sample_dst_data_dir=f"{flaw_dataset_dir}/sample/image/*"
# std data
std_src_data_dir = f"{com_dataset_dir}/train/image/*"

# sample dir
rec_src_sample_dir=f"{model_dir}/rec_sample/src_sample/"
rec_dst_sample_dir=f"{model_dir}/rec_sample/dst_sample/"

# initialize dir
os.makedirs(model_dir, exist_ok=True)
os.makedirs(rec_src_sample_dir, exist_ok=True)
os.makedirs(rec_dst_sample_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# log config
log_path=f"{model_dir}/log.txt"
init_log(log_path)
log(args)
log(time.asctime())

# train params
step, cur_epoch = 0,0
epoch_num = 50
batch_size = args.batch_size
log_interval = args.log_interval
save_interval = args.save_interval
sample_interval = args.sample_interval
learning_rate = 1e-4
fill_value = 1300/2300
warmup_iter = 5000
eps = 1/128**3
optim_dis_iter = 2
# model params
num_hiddens = 768
num_residual_hiddens = 256
num_residual_layers = 3
embedding_dim = 768

sample_flag = True
train_flag = True
val_flag = True
save_flag = True
sample_com_flag = True
sample_flaw_flag = True

# model config
model = DilatedAutoEncoder(in_channels=3,
                           num_hiddens=num_hiddens,
                           num_residual_layers=num_residual_layers,
                           num_residual_hiddens=num_residual_hiddens,
                           embedding_dim=embedding_dim)
dis = Gendis(in_channels=1,
                          num_hiddens=num_hiddens,
                          h=96, w=96, l=96)
# optim config
opt_model = optim.Adam(model.parameters(),
                        lr=learning_rate,
                        betas=(0.5,0.9),
                        eps=1e-8,
                        weight_decay=1e-5,
                        amsgrad=False)
opt_dis = optim.Adam(dis.parameters(),
                            lr=learning_rate,
                            betas=(0.5,0.9),
                            eps=1e-8,
                            weight_decay=1e-5,
                            amsgrad=False)

# load
if(os.path.exists(f"{model_dir}/newest_ckpt.pth")):
    log("loading ckpt...")
    ckpt = torch.load(f"{model_dir}/newest_ckpt.pth")
    step, cur_epoch, learning_rate = ckpt["step"], ckpt["epoch"], ckpt["learning_rate"]
    cur_epoch = cur_epoch+1   
    log(f"loaded step:{step} epoch:{cur_epoch} learning_rate:{learning_rate}")
if(os.path.exists(f"{model_dir}/newest_model.pth")):
    log("loading autoencoder...")
    model_state = torch.load(f"{model_dir}/newest_model.pth")
    model.load_state_dict(model_state)
    log("autoencoder loaded")
if(os.path.exists(f"{model_dir}/newest_dis.pth")):
    log("loading discriminator...")
    dis_state = torch.load(f"{model_dir}/newest_dis.pth")
    dis.load_state_dict(dis_state)
    log("global dis loaded")
if(os.path.exists(f"{model_dir}/newest_opt_model.pth")):
    log("loading optim_model...")
    opt_model_state = torch.load(f"{model_dir}/newest_opt_model.pth")
    opt_model.load_state_dict(opt_model_state)
if(os.path.exists(f"{model_dir}/newest_opt_dis.pth")):
    log("laoding optim_dis...")
    opt_dis_state = torch.load(f"{model_dir}/newest_opt_dis.pth")
    opt_dis.load_state_dict(opt_dis_state)
    

# datasets
# train
log("generating train datasets")
# inf src
inf_src_dataset = CompleteDataset3DFixMask(train_src_data_dir)
inf_src_dataloader = DataLoader(dataset=inf_src_dataset, batch_size=batch_size, shuffle=True)
inf_src_sampler = iter(repeater(inf_src_dataloader))
# inf dst
inf_dst_dataset = FlawDataset3DWithMask(train_dst_data_dir)
inf_dst_dataloader = DataLoader(dataset=inf_dst_dataset, batch_size=batch_size, shuffle=True)
inf_dst_sampler = iter(repeater(inf_dst_dataloader))
# inf std images
inf_std_dataset = CompleteDataset3DFixMask(std_src_data_dir)
inf_std_dataloader = DataLoader(dataset=inf_std_dataset, batch_size=batch_size, shuffle=True)
inf_std_sampler = iter(repeater(inf_std_dataloader))
# dst
train_dst_dataset = FlawDataset3DWithMask(train_dst_data_dir)
train_dst_dataloader = DataLoader(dataset=train_dst_dataset, batch_size=batch_size, shuffle=True)
log("")

# val
log("generating val datasets")
# src
val_src_dataset = CompleteDataset3DFixMask(val_src_data_dir)
val_src_dataloader = DataLoader(dataset=val_src_dataset, batch_size=batch_size)
# dst
val_dst_dataset = FlawDataset3DWithMask(val_dst_data_dir)
val_dst_dataloader = DataLoader(dataset=val_dst_dataset, batch_size=batch_size)
log("")

# sample
log("generating sample datasets")
# src
sample_src_dataset = CompleteDataset3DFixMask(sample_src_data_dir, ret_name=True)
sample_src_dataloader = DataLoader(dataset=sample_src_dataset, batch_size=1)
# dst
sample_dst_dataset = FlawDataset3DWithMask(sample_dst_data_dir, ret_name=True)
sample_dst_dataloader=DataLoader(dataset=sample_dst_dataset, batch_size=1)
log("")

# dataset summary
log(f"size train_dst_dataset:{len(train_dst_dataloader)} val_dst_dataset:{len(val_dst_dataloader)} "+\
    f"sample_dst_dataset:{len(sample_dst_dataloader)}")

# managers config
model_manager = ModelManager([model, dis])
metric_manager = MetricManager(root_dir=log_dir)

def infer_gen(gen:DilatedAutoEncoder,
              x,
              mask,
              bony_border_mask,
              ret_all=False,
              ret_dict=False):
    x_masked = x*(1-mask) + fill_value*mask
    input_x = torch.cat([x_masked, mask, bony_border_mask],dim=1)
    rec_x = gen.forward(input_x)
    con_rec_x = x*(1-mask)+rec_x*mask
    con_rec_x = con_rec_x.detach()
    if(ret_dict):
        return {
            "rec_x" : rec_x,
            "con_rec_x" : con_rec_x
        }
    elif(ret_all):
        return x_masked, rec_x, con_rec_x
    else:
        return con_rec_x

def update_dis(dis:Gendis,
               opt_dis:optim.Adam,
               fake_x,
               fake_mask,
               real_x,
               metric_manager:MetricManager,
               perfix=""):
    # optimize dis
    dis_loss = dis.cal_loss(real_x = real_x,
                            fake_x = fake_x,
                            mask = fake_mask,
                            metric_manager = metric_manager,
                            perfix = perfix)
    opt_dis.zero_grad()
    dis_loss.backward()
    opt_dis.step()

def update_gen(gen:DilatedAutoEncoder,
               dis:Gendis,
               opt_gen:optim.Adam,
               x,
               mask,
               bony_border_mask,
               metric_manager:MetricManager,
               cal_me_loss=True,
               cal_in_loss=True,
               cal_dis_loss=True,
               perfix=""):
    x_masked = x*(1-mask) + fill_value*mask
    input_x = torch.cat([x_masked, mask, bony_border_mask], dim=1)
    rec_x = gen.forward(input_x)
    
    l1_loss_alpha = 1.0
    g_loss = torch.zeros(size=[]).cuda()
    # ae loss
    g_ae_loss = l1_loss_alpha*torch.mean(torch.abs((x-rec_x)*(1-mask)))
    metric_manager.update("gen/g_ae_loss", g_ae_loss, perfix)
    g_loss += g_ae_loss
    
    # me loss
    g_me_loss = torch.zeros(size=[]).cuda()
    if(cal_me_loss):
        g_me_loss = l1_loss_alpha*torch.mean(torch.abs((x-rec_x)*mask))
    metric_manager.update(f"gen/g_me_loss",g_me_loss,perfix)
    g_loss += g_me_loss
    
    # in loss
    g_in_loss = torch.zeros(size=[]).cuda()
    if(cal_in_loss):
        g_in_loss = l1_loss_alpha*torch.mean(torch.abs((x-rec_x)*bony_border_mask))
    metric_manager.update("gen/g_in_loss", g_in_loss, perfix)
    g_loss += g_in_loss
    
    # sym loss
    sym_loss_alpha = 5
    inv_rec_x = get_register_inv_tensor(rec_x)
    g_sym_loss = sym_loss_alpha*torch.mean(torch.abs(inv_rec_x-rec_x))
    metric_manager.update("gen/g_sym_loss", g_sym_loss, perfix)
    g_loss += g_sym_loss
    
    # global dis loss
    wgan_loss_alpha = 5e-4
    g_dis_loss = torch.zeros(size=[]).cuda()
    if(cal_dis_loss):
        fake_pd = dis.forward(rec_x)
        g_dis_loss = wgan_loss_alpha*(-fake_pd.mean())
    metric_manager.update("gen/g_dis_loss", g_dis_loss, perfix)
    g_loss += g_dis_loss
    
    # total loss
    metric_manager.update("gen/g_total_loss", g_loss, perfix, new_line=True)
    
    # optmize
    if(opt_gen is not None):
        opt_gen.zero_grad()
        g_loss.backward()
        opt_gen.step()
    #del
    del x, mask, bony_border_mask, x_masked, input_x, rec_x, g_loss

def sample_gen(gen:DilatedAutoEncoder,
               x,
               mask,
               bony_border_mask,
               name,
               epoch,
               sample_dir
               ):
    x_masked, rec_x, con_rec_x = infer_gen(gen = gen,
                                           x = x,
                                           mask = mask,
                                           bony_border_mask = bony_border_mask,
                                           ret_all = True)
    inv_con_rec_x = get_register_inv_tensor(con_rec_x)
    # image
    image_path = f"{sample_dir}/{name}_origin.nii.gz"
    write_mha_tensor4D(x[0], image_path)
    # masked_x image
    masked_image_path = f"{sample_dir}/{name}_masked_{epoch}.nii.gz"
    write_mha_tensor4D(x_masked[0], masked_image_path)
    # mask image
    mask_path = f"{sample_dir}/{name}_mask_{epoch}.nii.gz"
    write_mha_tensor4D(mask[0], mask_path)
    # bony_border_mask image
    bony_border_mask_path = f"{sample_dir}/{name}_bony_border_mask_{epoch}.nii.gz"
    write_mha_tensor4D(bony_border_mask[0], bony_border_mask_path)
    # output_x image
    rec_path = f"{sample_dir}/{name}_rec_{epoch}.nii.gz"
    write_mha_tensor4D(rec_x[0], rec_path)
    # con_rec_x image
    con_rec_path = f"{sample_dir}/{name}_con_rec_{epoch}.nii.gz"
    write_mha_tensor4D(con_rec_x[0], con_rec_path)
    # inv_con_rec_x image
    inv_con_rec_path = f"{sample_dir}/{name}_inv_con_rec_{epoch}.nii.gz"
    write_mha_tensor4D(inv_con_rec_x[0], inv_con_rec_path)

    # l1 loss
    sample_loss = torch.mean(torch.abs(rec_x-x))
    log(f"\tsample name:{name} l1_loss:{sample_loss}")
    return sample_loss


# train
model_manager.train()
metric_manager.train()
model_manager.cuda()
log(f"start training step:{step} epoch:{cur_epoch} learning_rate:{learning_rate}")
for epoch in range(cur_epoch, epoch_num):
    log(f"epoch:{epoch}/{epoch_num}")
    start_time = time.time()
    learning_rate = 1e-4
    model.set_requires_grad(True)
    
    # sample
    if(sample_flag):
        model_manager.eval()
        if(epoch%sample_interval == 0):
            # sample stage
            # sample complete data
            if(sample_com_flag):
                log(f"[Sample com data]")
                src_sample_loss = 0
                for x, mask, bony_border_mask, name in tqdm(sample_src_dataloader):
                    with torch.no_grad():
                        src_sample_loss += sample_gen(
                            gen = model,
                            x = x,
                            mask = mask,
                            bony_border_mask = bony_border_mask,
                            name = name[0],
                            epoch = epoch,
                            sample_dir = rec_src_sample_dir)
                log(f"\tmean src_sample_loss:{src_sample_loss/len(sample_src_dataloader)}")

            # sample flaw data
            if(sample_flaw_flag):
                log(f"[Sample flaw data]")
                dst_sample_loss = 0
                for x, mask, bony_border_mask, name in tqdm(sample_dst_dataloader):
                    with torch.no_grad():
                        dst_sample_loss += sample_gen(
                            gen = model,
                            x = x,
                            mask = mask,
                            bony_border_mask = bony_border_mask,
                            name = name[0],
                            epoch = epoch,
                            sample_dir = rec_dst_sample_dir)
                log(f"\tmean dst_sample_loss:{dst_sample_loss/len(sample_dst_dataloader)}")

    # train
    if(train_flag):
        model_manager.train()
        metric_manager.train()
        for dst_x, dst_mask, dst_bony_border_mask in tqdm(train_dst_dataloader):
            step += 1
            # optimize dis
            model.set_requires_grad(False)
            dis.set_requires_grad(True)
            # optimize global
            for d_iter_idx in range(0, optim_dis_iter):
                # src
                iter_src_x, iter_src_mask, iter_src_bony_border_mask = next(inf_src_sampler)
                iter_src_real_x, iter_src_real_mask, iter_src_real_bony_border_mask = next(inf_std_sampler)
                iter_src_rec_x = infer_gen(gen = model,
                                           x = iter_src_x,
                                           mask = iter_src_mask,
                                           bony_border_mask = iter_src_bony_border_mask,
                                           ret_dict = True)["rec_x"]
                # dst
                iter_dst_x, iter_dst_mask, iter_dst_bony_border_mask = next(inf_dst_sampler)
                iter_dst_real_x, iter_dst_real_mask, iter_dst_real_bony_border_mask = next(inf_std_sampler)
                iter_dst_rec_x = infer_gen(gen = model,
                                           x = iter_dst_x,
                                           mask = iter_dst_mask,
                                           bony_border_mask = iter_dst_bony_border_mask,
                                           ret_dict = True)["rec_x"]

                # optimize global dis
                # optimize src
                update_dis(dis = dis,
                           opt_dis = opt_dis,
                           fake_x = iter_src_rec_x,
                           fake_mask = iter_src_mask,
                           real_x = iter_src_real_x,
                           metric_manager = metric_manager,
                           perfix = "src")
                # optimize dst
                update_dis(dis = dis,
                           opt_dis = opt_dis,
                           fake_x = iter_dst_rec_x,
                           fake_mask = iter_dst_mask,
                           real_x = iter_dst_real_x,
                           metric_manager = metric_manager,
                           perfix = "dst")
            
            # optimize gen
            model.encoder.set_requires_grad(True)
            if(step<=warmup_iter):
                model.decoder.set_requires_grad(True)
            dis.set_requires_grad(False)
            
            # optimize src, use global dis 
            src_x, src_mask, src_bony_border_mask = next(inf_src_sampler)
            update_gen(gen = model,
                       dis = dis,
                       opt_gen = opt_model,
                       x = src_x,
                       mask = src_mask,
                       bony_border_mask = src_bony_border_mask,
                       metric_manager = metric_manager,
                       cal_me_loss = True,
                       cal_in_loss = False,
                       cal_dis_loss = True,
                       perfix = "src")
            # optimize dst, use global dis
            update_gen(gen = model,
                       dis = dis,
                       opt_gen = opt_model,
                       x = dst_x,
                       mask = dst_mask,
                       bony_border_mask = dst_bony_border_mask,
                       metric_manager = metric_manager,
                       cal_me_loss = False,
                       cal_in_loss = True,
                       cal_dis_loss = True,
                       perfix="dst")

            #log
            if(step%log_interval == 0):
                log(f"[Train][{get_cur_time_str()}] epoch:{epoch} step:{step} learning_rate:{learning_rate} \n{metric_manager.report_log()}\n")
        
    train_end_time = time.time()

    # val
    if(val_flag):
        model_manager.eval()
        metric_manager.eval()
        # src
        for src_x, src_mask, src_bony_border_mask in tqdm(val_src_dataloader):
            with torch.no_grad():
                update_gen(gen = model,
                           dis = dis,
                           opt_gen = None,
                           x = src_x,
                           mask = src_mask,
                           bony_border_mask = src_bony_border_mask,
                           metric_manager = metric_manager,
                           cal_me_loss = True,
                           cal_in_loss = False,
                           cal_dis_loss = True,
                           perfix = "src")

        # dst  
        for dst_x, dst_mask, dst_bony_border_mask in tqdm(val_dst_dataloader):
            with torch.no_grad():
                update_gen(gen = model,
                           dis = dis,
                           opt_gen = None,
                           x = dst_x,
                           mask = dst_mask,
                           bony_border_mask = dst_bony_border_mask,
                           metric_manager = metric_manager,
                           cal_me_loss = False,
                           cal_in_loss = True,
                           cal_dis_loss = True,
                           perfix = "dst")

    val_end_time = time.time()

    # summary
    log(f"[Train Summary][{get_cur_time_str()}] epoch:{epoch} step:{step} time:{(train_end_time-start_time)/3600} learning_rate:{learning_rate} \n{metric_manager.report_train()}\n")
    
    log(f"[Val Summary][{get_cur_time_str()}] epoch:{epoch} step:{step} time:{(val_end_time-train_end_time)/3600} learning_rate:{learning_rate} \n{metric_manager.report_val()}\n")
    
    # draw
    metric_manager.draw()

    # save
    if(save_flag):
        save_flag_list = ["newest"]
        if(epoch % save_interval == 0):
            save_flag_list.append(str(epoch))
        for save_flag in save_flag_list:
            ckpt = {}
            ckpt["step"], ckpt["epoch"], ckpt["learning_rate"] = step, epoch, learning_rate
            torch.save(ckpt, f"{model_dir}/{save_flag}_ckpt.pth")
            torch.save(model.state_dict(), f"{model_dir}/{save_flag}_model.pth")
            torch.save(dis.state_dict(), f"{model_dir}/{save_flag}_dis.pth")
            torch.save(opt_model.state_dict(), f"{model_dir}/{save_flag}_opt_model.pth")
            torch.save(opt_dis.state_dict(), f"{model_dir}/{save_flag}_opt_dis.pth")
        log("ckpt saved!")
    
        


    
        

