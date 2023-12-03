#version 410 core
out vec4 FragColor;

in vec2 TexCoords;

struct DirLight {
    vec3 direction;
	
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

uniform float Shininess;
uniform vec3 viewPos;
uniform DirLight dirLight;
uniform vec3 Diffuse;
uniform float Specular;

uniform sampler2D gPosition;
uniform sampler2D gNormal;

// function prototypes
vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir, vec3 Diffuse, float Specular);

void main()
{    

    // properties
    vec3 FragPos = texture(gPosition, TexCoords).rgb;
    vec3 Normal = texture(gNormal, TexCoords).rgb;

    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(viewPos - FragPos);

    
    // directional lighting

    //FragColor = vec4(1, 0, 0, 1);
    FragColor = vec4(CalcDirLight(dirLight, norm, viewDir, Diffuse, Specular), 1.0);

}

// calculates the color when using a directional light.
vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir, vec3 Diffuse, float Specular)
{
    vec3 lightDir = normalize(-light.direction);
    // diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), Shininess);
    // combine results
    vec3 ambient = light.ambient * Diffuse;
    vec3 diffuse = light.diffuse * diff * Diffuse;
    vec3 specular = light.specular * spec * Specular;

    return (ambient + diffuse + specular);
}
