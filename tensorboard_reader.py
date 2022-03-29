from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

for event in EventFileLoader('C:\\Users\\Public\\Desktop\\Tesis\\Models\\Quimera\\logs\\fit\\20220315-000523\\events.out.tfevents.1647313523.turing.3139684.9331.v2').Load():
    print(len(event))