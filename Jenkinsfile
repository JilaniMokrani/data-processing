pipeline {
    agent none
    stages {
        stage('data-preprocessing') {
            agent {
                docker {
                    image "jilani95/data-preprocessing"
                    args "-v /home/Desktop/Project/RawData:/data -v /home/Desktop/Project/OutputData/:/output"
                }
            }
            steps {
                print 'hello world'
            }
        }
    }
}