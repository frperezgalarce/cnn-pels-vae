import src.cnn as cnn
import src.bgmm as bgmm 
import src.fit_regressor as reg

def main():
    reg.main()
    bgmm.train_and_save()
    cnn.run_cnn()

if __name__ == "__main__":
    main()
